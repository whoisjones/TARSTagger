from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification

import torch

from src.corpora import load_corpus, split_dataset
from src.models.metrics import compute_metrics
from src.utils import tokenize_and_align_labels


def baseline(args, run):

    device = f"cuda{':' + args.cuda_devices}" if args.cuda and torch.cuda.is_available() else "cpu"
    output_dir = f"{args.output_dir}/run{run}"

    # load dataset
    dataset, tags, index2tag, tag2index = load_corpus(args.corpus)

    # model
    config = AutoConfig.from_pretrained(args.language_model, num_labels=tags.num_classes,
                                        id2label=index2tag, label2id=tag2index)
    model = AutoModelForTokenClassification.from_pretrained(args.language_model, config=config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # preprocessing
    tokenized_dataset = dataset.map(lambda p: tokenize_and_align_labels(p, tokenizer), batched=True)
    train_dataset, validation_dataset, test_dataset = split_dataset(tokenized_dataset)

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=float(args.lr),
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
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
    if args.save_model:
        trainer.save_model()

    metrics["train_samples"] = len(dataset["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(dataset["validation"])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    if args.final_test:
        predictions, labels, metrics = trainer.predict(test_dataset, metric_key_prefix="predict")
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
