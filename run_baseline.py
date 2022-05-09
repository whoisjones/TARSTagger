import argparse
import logging

import transformers
from transformers import AutoConfig, AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification

import datasets
from datasets import load_dataset

import torch
import numpy as np
from seqeval.metrics import classification_report, f1_score

from model import TARSTagger
from data import tokenize_and_align_labels

logger = logging.getLogger(__name__)


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

def get_f1_score(trainer, dataset):
    return trainer.predict(dataset).metrics["test_f1"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="xlm-roberta-large")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=10)

    args = parser.parse_args()

    log_level = logging.INFO
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = load_dataset("conll2003")
    tags = dataset["train"].features["ner_tags"].feature
    index2tag = {idx: tag for idx, tag in enumerate(tags.names)}
    tag2index = {tag: idx for idx, tag in enumerate(tags.names)}

    config = AutoConfig.from_pretrained(args.model, num_labels=tags.num_classes,
                                        id2label=index2tag, label2id=tag2index)

    model = TARSTagger.from_pretrained(args.model, config=config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    tokenized_dataset = dataset.map(lambda p: tokenize_and_align_labels(p, tokenizer), batched=True,
                                    remove_columns=["tokens", "pos_tags", "chunk_tags", "ner_tags"])

    logging_steps = len(tokenized_dataset["train"]) // args.batch_size

    path = "resouces/tars/run1/"
    training_arguments = TrainingArguments(
        output_dir="resouces/tars/run1",
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=logging_steps,
        weight_decay=0.01
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    trainer.save_model(f"{path}/final")

if __name__ == "__main__":
    main()
