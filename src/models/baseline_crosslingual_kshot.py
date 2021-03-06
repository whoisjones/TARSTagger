from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification
import json
import random
import torch
import numpy as np
import os
from prettytable import PrettyTable
from seqeval.metrics import classification_report, f1_score

from src.corpora import load_corpus, split_dataset, load_label_id_mapping
from src.utils import tokenize_and_align_labels, k_shot_sampling


def eval_crossling_baseline_kshot(args, run):

    for model_run in os.listdir(args.language_model):

        for k in [1,2,4,8,16,32]:

            random.seed(run)
            np.random.seed(run)
            torch.manual_seed(run)
            torch.cuda.manual_seed_all(run)

            # set cuda device
            device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

            output_dir = f"cross_lingual_baseline_kshot/{args.output_dir}{model_run[-1]}_{k}shot/run{run}"
            language_model = f"{args.language_model}/{model_run}"

            tokenizer = AutoTokenizer.from_pretrained(language_model)
            data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

            dataset, tags, index2tag, tag2index = load_corpus(args.corpus)

            tokenized_dataset = dataset.map(lambda p: tokenize_and_align_labels(p, tokenizer), batched=True)
            train_dataset, validation_dataset, test_dataset = split_dataset(tokenized_dataset)

            label_id_mapping_train = load_label_id_mapping(train_dataset, index2tag)
            label_id_mapping_validation = load_label_id_mapping(validation_dataset, index2tag)

            model = AutoModelForTokenClassification.from_pretrained(language_model).to(device)
            few_shot_classifier = torch.nn.Linear(in_features=model.classifier.in_features,
                                                  out_features=tags.num_classes).to(device)
            if args.reuse_decoder_weights:
                _, _, reuse_idx2tag, _ = load_corpus(args.reuse_corpus_for_weights)
                with torch.no_grad():
                    for _idx, _tag in reuse_idx2tag.items():
                        if _tag in tag2index:
                            few_shot_classifier.weight[tag2index[_tag]] = model.classifier.weight[_idx]
                        elif _tag == "B-PER":
                            few_shot_classifier.weight[tag2index["B-PERSON"]] = model.classifier.weight[_idx]
                        elif _tag == "I-PER":
                            few_shot_classifier.weight[tag2index["I-PERSON"]] = model.classifier.weight[_idx]
            model.classifier = few_shot_classifier.to(device)
            model.num_labels = tags.num_classes

            train_kshot_indices, train_label_count = k_shot_sampling(k=k, mapping=label_id_mapping_train, seed=run, mode=args.sampling_mode)
            validation_kshot_indices, val_label_count = k_shot_sampling(k=k, mapping=label_id_mapping_validation, seed=run, mode=args.sampling_mode)

            train_dataset, validation_dataset, test_dataset = split_dataset(tokenized_dataset)
            train_dataset = train_dataset.filter(lambda example: example["id"] in train_kshot_indices)
            validation_dataset = validation_dataset.filter(lambda example: example["id"] in validation_kshot_indices)

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

            table = PrettyTable(["Parameter", "Value"])

            for parameter, value in args.__dict__.items():
                table.add_row([parameter, value])

            with open(f"{output_dir}/kshot_config.txt", "w+") as f:
                f.write(str(table))

            train_label_count["indices"] = train_kshot_indices
            train_label_count["number examples"] = len(set(train_kshot_indices))
            with open(f"{output_dir}/support_set.json", "w+") as f:
                json.dump(train_label_count, f, indent=2)

            val_label_count["indices"] = validation_kshot_indices
            val_label_count["number examples"] = len(set(validation_kshot_indices))
            with open(f"{output_dir}/validation_set.json", "w+") as f:
                json.dump(val_label_count, f, indent=2)
