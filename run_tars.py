import argparse

from transformers import AutoModelForTokenClassification
from transformers import AutoConfig, AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification

from datasets import load_dataset

import torch
from torch.utils.data import DataLoader
import numpy as np
from seqeval.metrics import classification_report, f1_score

from data import make_tars_dataset

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main(args):
    # set cuda device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # load dataset
    dataset = load_dataset("conll2003")

    if "train" in dataset:
        train_dataset = load_dataset("conll2003", split="train[:4]")
    if "validation" in dataset:
        val_dataset = load_dataset("conll2003", split="train[:4]")
    if "test" in dataset:
        test_dataset = load_dataset("conll2003", split="train[:4]")

    original_tags = train_dataset.features["ner_tags"].feature
    original_index2original_tag = {idx: tag for idx, tag in enumerate(original_tags.names)}

    tars_tag2tars_label = {"O": "O", "B-PER": "person", "I-PER": "person", "B-ORG": "organization", "I-ORG": "organization",
                "B-LOC": "location", "I-LOC": "location", "B-MISC": "miscellaneous",
                "I-MISC": "miscellaneous"}
    tars_label2tars_tag = {}
    for k, v in tars_tag2tars_label.items():
        tars_label2tars_tag[v] = tars_label2tars_tag.get(v, []) + [k]

    tars_prediction_head = {'O': 0, 'B-': 1, 'I-': 2}
    inv_tars_prediction_head = {v: k for k, v in tars_prediction_head.items()}

    train_dataset = make_tars_dataset(dataset=train_dataset,
                                      tokenizer=tokenizer,
                                      index2tag=original_index2original_tag,
                                      tag2tars=tars_tag2tars_label,
                                      tars_head=tars_prediction_head,
                                      num_negatives="one")

    val_dataset = make_tars_dataset(dataset=val_dataset,
                                    tokenizer=tokenizer,
                                    index2tag=original_index2original_tag,
                                    tag2tars=tars_tag2tars_label,
                                    tars_head=tars_prediction_head,
                                    num_negatives="one")

    test_dataset = make_tars_dataset(dataset=test_dataset,
                                     tokenizer=tokenizer,
                                     index2tag=original_index2original_tag,
                                     tag2tars=tars_tag2tars_label,
                                     tars_head=tars_prediction_head,
                                     num_negatives="all")

    config = AutoConfig.from_pretrained(args.model, num_labels=len(tars_prediction_head),
                                        id2label=inv_tars_prediction_head, label2id=tars_prediction_head)

    model = AutoModelForTokenClassification.from_pretrained(args.model, config=config).to(device)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
    )

    def align_predictions(predictions, label_ids):
        preds = np.argmax(predictions, axis=2)
        batch_size, seq_len = preds.shape
        labels_list, preds_list = [], []

        for batch_idx in range(batch_size):
            example_labels, example_preds = [], []
            for seq_idx in range(seq_len):
                if label_ids[batch_idx, seq_idx] != -100:
                    example_labels.append(inv_tars_prediction_head[label_ids[batch_idx][seq_idx]])
                    example_preds.append(inv_tars_prediction_head[preds[batch_idx][seq_idx]])

            labels_list.append(example_labels)
            preds_list.append(example_preds)

        return preds_list, labels_list

    def compute_metrics(eval_pred):
        y_pred, y_true = align_predictions(eval_pred.predictions, eval_pred.label_ids)
        return {"classification_report": classification_report(y_true, y_pred),
                "f1": f1_score(y_true, y_pred)}

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.save_model()

    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(val_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    test_dataloader = DataLoader(test_dataset.remove_columns(["id", "ner_tags", "tars_tags", "tars_label_length", "tars_labels"]),
                                 shuffle=False, collate_fn=data_collator, batch_size=16)

    with torch.no_grad():
        outputs = []
        for batch in test_dataloader:
            outputs.extend(model(**{k: v.to(device) for k, v in batch.items()}).logits)

    def extract_aligned_logits(row):
        logits_list = []

        labels = row["labels"]
        for label_idx in range(labels.shape[0]):
            if labels[label_idx] != -100:
                logits_list.append(outputs[row.name][label_idx])

        if logits_list:
            return torch.stack(logits_list)
        else:
            return []

    df = test_dataset.to_pandas()
    df["logits"] = df.apply(lambda row: extract_aligned_logits(row), axis=1)
    df = df[df["logits"].map(len) > 0]
    df = df.groupby(["id"])

    def to_original_tag(tars_tag, tars_label):
        original_tag_prefix = inv_tars_prediction_head.get(tars_tag)
        if original_tag_prefix != "O":
            original_tag = [tag for tag in tars_label2tars_tag.get(tars_label) if original_tag_prefix in tag]
            assert len(original_tag) == 1
            return original_tag.pop()
        else:
            return original_tag_prefix

    predictions, labels = [], []
    for idx, tars_predictions in df:
        logits_per_tars_label = torch.stack(tars_predictions["logits"].to_list()).cpu().detach().numpy()
        pred_tars_labels = np.argmax(logits_per_tars_label, axis=2)
        score_tars_label = np.max(logits_per_tars_label, axis=2)
        current_preds = []
        for col_idx in range(pred_tars_labels.shape[1]):
            if not pred_tars_labels[:, col_idx].any():
                current_preds.append(
                    to_original_tag(
                        tars_tag=0,
                        tars_label=tars_predictions["tars_labels"].iloc[0]
                    )
                )
            else:
                nonzero_entries = np.nonzero(pred_tars_labels[:, col_idx])
                if nonzero_entries[0].shape[0] == 1:
                    tars_tag = pred_tars_labels[:, col_idx][nonzero_entries][0]
                    tars_label = tars_predictions["tars_labels"].iloc[nonzero_entries][0]
                else:
                    id_from_max_score = nonzero_entries[0][np.argmax(score_tars_label[:, col_idx][nonzero_entries])]
                    tars_tag = pred_tars_labels[id_from_max_score, col_idx]
                    tars_label = tars_predictions["tars_labels"].iloc[id_from_max_score]

                current_preds.append(
                    to_original_tag(
                        tars_tag=tars_tag,
                        tars_label=tars_label
                    )
                )

        assert all((element == tars_predictions["ner_tags"].to_list()[0]).all() for element in tars_predictions["ner_tags"].to_list())
        current_labels = [original_index2original_tag.get(x) for x in tars_predictions["ner_tags"].iloc[0].tolist()]

        predictions.append(current_preds)
        labels.append(current_labels)

    results = classification_report(labels, predictions)

    print(results)

    with open("results.txt", "w+") as f:
        f.write(results)

if __name__ == "__main__":

    # parser training arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="xlm-roberta-large")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    main(args)
