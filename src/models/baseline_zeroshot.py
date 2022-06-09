import os

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification

import torch
from seqeval.metrics import classification_report

from src.corpora import load_corpus, split_dataset
from src.utils import tokenize_and_align_labels


def baseline_zeroshot(args, run):

    # set cuda device
    device = f"cuda{':' + args.cuda_devices}" if args.cuda and torch.cuda.is_available() else "cpu"
    output_dir = f"{args.output_dir}_0shot/run{run}"

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    dataset, tags, index2tag, tag2index = load_corpus(args.corpus)

    tokenized_dataset = dataset.map(lambda p: tokenize_and_align_labels(p, tokenizer), batched=True)
    train_dataset, validation_dataset, test_dataset = split_dataset(tokenized_dataset)

    model = AutoModelForTokenClassification.from_pretrained(args.language_model).to(device)
    few_shot_classifier = torch.nn.Linear(in_features=model.classifier.in_features,
                                          out_features=tags.num_classes).to(device)
    if args.reuse_decoder_weights:
        _, _, reuse_idx2tag, _ = load_corpus(args.reuse_corpus_for_weights)
        with torch.no_grad():
            for _idx, _tag in reuse_idx2tag.items():
                if _tag in tag2index:
                    few_shot_classifier.weight[tag2index[_tag]] = model.classifier.weight[_idx]
    model.classifier = few_shot_classifier.to(device)
    model.num_labels = tags.num_classes
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    def get_test_dataloader(test_dataset):
        test_dataloader = DataLoader(
            test_dataset.remove_columns(set(test_dataset.column_names) - set(["input_ids", "attention_mask", "labels"])),
            collate_fn=data_collator,
            shuffle=False,
            batch_size=args.eval_batch_size
        )
        return test_dataloader

    def get_logits(test_dataloader):
        with torch.no_grad():
            outputs = []
            for batch in tqdm(test_dataloader):
                outputs.extend(model(**{k: v.to(device) for k, v in batch.items()}).logits)
        return outputs

    test_dataloader = get_test_dataloader(test_dataset)
    outputs = get_logits(test_dataloader)

    preds, labels = [], []
    for logits, inputs in zip(outputs, test_dataset):
        max_logit_preds = logits.argmax(dim=1).detach().cpu().numpy()
        curr_preds, curr_labels = [], []
        for max_logit_pred, label in zip(max_logit_preds, inputs["labels"]):
            if label != -100:
                curr_preds.append(index2tag[max_logit_pred])
                curr_labels.append(index2tag[label])
        preds.append(curr_preds)
        labels.append(curr_labels)

    os.makedirs(output_dir)
    results = classification_report(labels, preds)
    with open(f"{output_dir}/results.txt", "w+") as f:
        f.write(results)
