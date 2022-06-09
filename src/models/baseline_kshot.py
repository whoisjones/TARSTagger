from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification

import torch

from src.corpora import load_corpus, split_dataset, load_label_id_mapping
from src.models.metrics import compute_metrics
from src.utils import tokenize_and_align_labels, k_shot_sampling


def baseline_kshot(args, run):

    # set cuda device
    device = f"cuda{':' + args.cuda_devices}" if args.cuda and torch.cuda.is_available() else "cpu"
    output_dir = f"{args.output_dir}_{args.k}shot/run{run}"

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    dataset, tags, index2tag, tag2index = load_corpus(args.corpus)

    tokenized_dataset = dataset.map(lambda p: tokenize_and_align_labels(p, tokenizer), batched=True)
    train_dataset, validation_dataset, test_dataset = split_dataset(tokenized_dataset)

    label_id_mapping_train = load_label_id_mapping(train_dataset)
    label_id_mapping_validation = load_label_id_mapping(validation_dataset)

    model = AutoModelForTokenClassification.from_pretrained(args.pretrained_model_path).to(device)
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

    train_kshot_indices = k_shot_sampling(k=args.k, mapping=label_id_mapping_train, seed=run)
    validation_kshot_indices = k_shot_sampling(k=args.k, mapping=label_id_mapping_validation, seed=run)

    train_dataset, validation_dataset, test_dataset = split_dataset(tokenized_dataset)
    train_dataset = train_dataset.select(train_kshot_indices)
    validation_dataset = validation_dataset.select(validation_kshot_indices)

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=float(args.lr),
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
