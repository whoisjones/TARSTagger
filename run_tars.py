import argparse
import random

from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification

from datasets import load_dataset

import torch
import numpy as np
from seqeval.metrics import classification_report, f1_score

from data import tokenize_and_align_labels, make_tars_dataset


def main():
    # parser training arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="xlm-roberta-large")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    # set cuda device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # load dataset
    dataset = load_dataset("conll2003")

    if "train" in dataset:
        train_dataset = dataset["train"]
    if "validation" in dataset:
        val_dataset = dataset["train"]
    if "test" in dataset:
        test_dataset = dataset["test"]

    tag2tars = {"O": "O", "B-PER": "person", "I-PER": "person", "B-ORG": "organization", "I-ORG": "organization",
                "B-LOC": "location", "I-LOC": "location", "B-MISC": "miscellaneous",
                "I-MISC": "miscellaneous"}
    tars_head = {'O': 0, 'B-': 1, 'I-': 2}

    train_dataset = make_tars_dataset(dataset=train_dataset,
                                      tokenizer=tokenizer,
                                      tag2tars=tag2tars,
                                      tars_head=tars_head)

    val_dataset = make_tars_dataset(dataset=val_dataset,
                                    tokenizer=tokenizer,
                                    tag2tars=tag2tars,
                                    tars_head=tars_head)

    test_dataset = make_tars_dataset(dataset=test_dataset,
                                     tokenizer=tokenizer,
                                     tag2tars=tag2tars,
                                     tars_head=tars_head)

    index2tag = {v: k for k, v in tars_head.items()}

    config = AutoConfig.from_pretrained(args.model, num_labels=len(tars_head),
                                        id2label=index2tag, label2id=tars_head)

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
                    example_labels.append(index2tag[label_ids[batch_idx][seq_idx]])
                    example_preds.append(index2tag[preds[batch_idx][seq_idx]])

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

    predictions, labels, metrics = trainer.predict(test_dataset, metric_key_prefix="predict")
    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)


if __name__ == "__main__":
    main()
