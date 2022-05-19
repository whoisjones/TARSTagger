import argparse

from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification


import torch
import numpy as np
from seqeval.metrics import classification_report, f1_score

from corpora import load_corpus, split_dataset
from preprocessing import tokenize_and_align_labels


def main(args):

    # set cuda device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = AutoModelForTokenClassification.from_pretrained(args.pretrained_model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    dataset, tags, index2tag, tag2index = load_corpus(args.corpus)
    tokenized_dataset = dataset.map(lambda p: tokenize_and_align_labels(p, tokenizer), batched=True)
    train_dataset, validation_dataset, test_dataset = split_dataset(tokenized_dataset)

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
        eval_dataset=validation_dataset,
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
    metrics["eval_samples"] = len(validation_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

if __name__ == "__main__":

    # parser training arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="xlm-roberta-large")
    parser.add_argument("--corpus", type=str, default="conll")
    parser.add_argument("--pretrained_model_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    main(args)
