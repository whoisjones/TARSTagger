import argparse
import random

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification

import torch
import numpy as np
from seqeval.metrics import classification_report, f1_score

from corpora import load_corpus, split_dataset, load_label_id_mapping
from preprocessing import tokenize_and_align_labels, k_shot_sampling

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(args):

    # set cuda device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    dataset, tags, index2tag, tag2index = load_corpus(args.corpus)
    conll_data, conll_tags, conll_idx2tag, conll_tag2idx = load_corpus("conll")
    tokenized_dataset = dataset.map(lambda p: tokenize_and_align_labels(p, tokenizer), batched=True)
    train_dataset, validation_dataset, test_dataset = split_dataset(tokenized_dataset)
    label_id_mapping_train = load_label_id_mapping(train_dataset)
    label_id_mapping_validation = load_label_id_mapping(validation_dataset)

    def align_predictions(predictions, label_ids):
        preds = np.argmax(predictions, axis=2)
        batch_size, seq_len = preds.shape
        labels_list, preds_list = [], []

        for batch_idx in range(batch_size):
            example_labels, example_preds = [], []
            for seq_idx in range(seq_len):
                if label_ids[batch_idx, seq_idx] != -100:
                    example_labels.append(index2tag[label_ids[batch_idx][seq_idx]])
                    example_preds.append(index2tag[preds[batch_idx][seq_idx]])

            labels_list.append(example_labels)
            preds_list.append(example_preds)

        return preds_list, labels_list

    def compute_metrics(eval_pred):
        y_pred, y_true = align_predictions(eval_pred.predictions, eval_pred.label_ids)
        return {"classification_report": classification_report(y_true, y_pred),
                "f1": f1_score(y_true, y_pred)}

    for k in args.k:

        if k == 0:

            output_dir = f"resources/{args.corpus}_baseline_{k}shot"

            model = AutoModelForTokenClassification.from_pretrained(args.pretrained_model_path).to(device)
            few_shot_classifier = torch.nn.Linear(in_features=model.classifier.in_features,
                                                  out_features=tags.num_classes).to(device)
            with torch.no_grad():
                for conll_idx, conll_tag in conll_idx2tag.items():
                    if conll_tag in tag2index:
                        few_shot_classifier.weight[tag2index[conll_tag]] = model.classifier.weight[conll_idx]
            model.classifier = few_shot_classifier.to(device)
            model.num_labels = tags.num_classes

            data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

            def get_test_dataloader(test_dataset):
                test_dataloader = DataLoader(
                    test_dataset.remove_columns(set(test_dataset.column_names) - set(["input_ids", "attention_mask", "labels"])),
                    collate_fn=data_collator,
                    shuffle=False,
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

        else:
            for run in range(args.seed):

                output_dir = f"resources/{args.corpus}_baseline_{k}shot/run{run}"

                model = AutoModelForTokenClassification.from_pretrained(args.pretrained_model_path).to(device)
                few_shot_classifier = torch.nn.Linear(in_features=model.classifier.in_features,
                                                      out_features=tags.num_classes).to(device)
                with torch.no_grad():
                    for conll_idx, conll_tag in conll_idx2tag.items():
                        if conll_tag in tag2index:
                            few_shot_classifier.weight[tag2index[conll_tag]] = model.classifier.weight[conll_idx]
                model.classifier = few_shot_classifier.to(device)
                model.num_labels = tags.num_classes

                train_kshot_indices = k_shot_sampling(k=k, mapping=label_id_mapping_train, seed=run)
                validation_kshot_indices = k_shot_sampling(k=k, mapping=label_id_mapping_validation, seed=run)

                train_dataset, validation_dataset, test_dataset = split_dataset(tokenized_dataset)
                train_dataset = train_dataset.select(train_kshot_indices)
                validation_dataset = validation_dataset.select(validation_kshot_indices)

                training_arguments = TrainingArguments(
                    output_dir=output_dir,
                    evaluation_strategy="epoch",
                    save_strategy="no",
                    learning_rate=args.lr,
                    per_device_train_batch_size=args.batch_size,
                    per_device_eval_batch_size=args.batch_size,
                    num_train_epochs=args.epochs,
                )

                trainer = Trainer(
                    model=model,
                    args=training_arguments,
                    train_dataset=train_dataset,
                    eval_dataset=validation_dataset,
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    compute_metrics=compute_metrics
                )

                train_result = trainer.train()
                metrics = train_result.metrics

                metrics["train_samples"] = len(train_dataset)
                trainer.log_metrics("train", metrics)
                trainer.save_metrics("train", metrics)
                trainer.save_state()

                metrics = trainer.evaluate()
                metrics["eval_samples"] = len(validation_dataset)
                trainer.log_metrics("eval", metrics)
                trainer.save_metrics("eval", metrics)

                predictions, labels, metrics = trainer.predict(test_dataset, metric_key_prefix="predict")
                trainer.log_metrics("predict", metrics)
                trainer.save_metrics("predict", metrics)

if __name__ == "__main__":

    # parser training arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="xlm-roberta-large")
    parser.add_argument("--corpus", type=str, default="conll")
    parser.add_argument("--pretrained_model_path", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--k", type=list, default=[1,2,4,8,16,32,64])
    args = parser.parse_args()

    main(args)
