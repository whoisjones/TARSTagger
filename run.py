import argparse

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification
from datasets import load_dataset

from model import TARSTagger
from data import tokenize_and_align_labels
from metric import compute_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="xlm-roberta-large")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--existing_model_path", type=str, default=None)

    args = parser.parse_args()

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

    training_arguments = TrainingArguments(
        output_dir="resouces/tars/run1",
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
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

    trainer.predict(tokenized_dataset["test"])

