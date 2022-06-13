import argparse
import os
from tqdm import tqdm

from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification


import torch
from torch.utils.data import DataLoader
import numpy as np
from seqeval.metrics import classification_report, f1_score

from src.corpora import load_corpus, split_dataset, load_label_id_mapping
from src.utils.tars_format import make_tars_datasets, load_tars_label_mapping
from src.utils import k_shot_sampling


def tars_kshot(args, run):

    # set cuda device
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    output_dir = f"{args.output_dir}_{args.k}shot/run{run}"

    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    model = AutoModelForTokenClassification.from_pretrained(args.language_model).to(device)

    dataset, tags, index2tag, tag2index = load_corpus(args.corpus)
    train_dataset, validation_dataset, test_dataset = split_dataset(dataset)
    tars_tag2id = {'O': 0, 'B-': 1, 'I-': 2}
    tars_id2tag = {v: k for k, v in tars_tag2id.items()}
    org_tag2tars_label, tars_label2org_tag = load_tars_label_mapping(tags)

    label_id_mapping_train = load_label_id_mapping(train_dataset, index2tag)
    label_id_mapping_validation = load_label_id_mapping(validation_dataset, index2tag)

    train_kshot_indices = k_shot_sampling(k=args.k, mapping=label_id_mapping_train, seed=run, mode=args.sampling_mode)
    validation_kshot_indices = k_shot_sampling(k=args.k, mapping=label_id_mapping_validation, seed=run, mode=args.sampling_mode)

    kshot_train_dataset = train_dataset.filter(lambda example: example["id"] in train_kshot_indices)
    kshot_validation_dataset = validation_dataset.filter(lambda example: example["id"] in validation_kshot_indices)

    kshot_train_dataset, kshot_validation_dataset, test_dataset = make_tars_datasets(
        datasets=[kshot_train_dataset, kshot_validation_dataset, test_dataset],
        tokenizer=tokenizer,
        index2tag=index2tag,
        org_tag2tars_label=org_tag2tars_label,
        tars_tag2id=tars_tag2id,
        num_negatives=args.num_negatives
    )

    def align_predictions(predictions, label_ids):
        preds = np.argmax(predictions, axis=2)
        batch_size, seq_len = preds.shape
        labels_list, preds_list = [], []

        for batch_idx in range(batch_size):
            example_labels, example_preds = [], []
            for seq_idx in range(seq_len):
                if label_ids[batch_idx, seq_idx] != -100:
                    example_labels.append(tars_id2tag[label_ids[batch_idx][seq_idx]])
                    example_preds.append(tars_id2tag[preds[batch_idx][seq_idx]])

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
        train_dataset=kshot_train_dataset,
        eval_dataset=kshot_validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    train_result = trainer.train()
    metrics = train_result.metrics

    metrics["train_samples"] = len(kshot_train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(kshot_validation_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    def get_test_dataloader(test_dataset):
        test_dataloader = DataLoader(
            test_dataset.remove_columns(["id", "ner_tags", "tars_tags", "tars_label_length", "tars_labels"]),
            shuffle=False,
            collate_fn=data_collator,
            batch_size=args.eval_batch_size
        )
        return test_dataloader

    def get_logits(test_dataloader):
        with torch.no_grad():
            outputs = []
            for batch in tqdm(test_dataloader):
                outputs.extend(model(**{k: v.to(device) for k, v in batch.items()}).logits)
        return outputs

    def extract_aligned_logits(row):
        logits_list = []

        labels = row["labels"]
        for label_idx in range(labels.shape[0]):
            if labels[label_idx] != -100:
                logits_list.append(logits[row.name][label_idx])

        if logits_list:
            return torch.stack(logits_list)
        else:
            return []

    def group_predictions(test_dataset):
        df = test_dataset.to_pandas()
        df["logits"] = df.apply(lambda row: extract_aligned_logits(row), axis=1)
        df = df[df["logits"].map(len) > 0]
        df = df.groupby(["id"])
        return df

    def to_original_tag(tars_tag, tars_label):
        original_tag_prefix = tars_id2tag.get(tars_tag)
        if original_tag_prefix != "O":
            original_tag = [tag for tag in tars_label2org_tag.get(tars_label) if original_tag_prefix in tag]
            assert len(original_tag) == 1
            return original_tag.pop()
        else:
            return original_tag_prefix

    def evaluate(predictions_df):
        predictions, labels = [], []
        for idx, tars_predictions in predictions_df:
            raw_preds = tars_predictions["logits"].to_list()
            truncate_upper_bound = min([p.shape[0] for p in raw_preds])
            truncated_preds = [p[:truncate_upper_bound] for p in raw_preds]
            logits_per_tars_label = torch.stack(truncated_preds).cpu().detach().numpy()
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
                        tars_label = tars_predictions["tars_labels"].iloc[nonzero_entries].iloc[0]
                    else:
                        id_from_max_score = nonzero_entries[0][
                            np.argmax(score_tars_label[:, col_idx][nonzero_entries])]
                        tars_tag = pred_tars_labels[id_from_max_score, col_idx]
                        tars_label = tars_predictions["tars_labels"].iloc[id_from_max_score]

                    current_preds.append(
                        to_original_tag(
                            tars_tag=tars_tag,
                            tars_label=tars_label
                        )
                    )

            assert all((element == tars_predictions["ner_tags"].to_list()[0]).all() for element in
                       tars_predictions["ner_tags"].to_list())
            current_labels = [index2tag.get(x) for x in tars_predictions["ner_tags"].iloc[0].tolist()[:truncate_upper_bound]]

            predictions.append(current_preds)
            labels.append(current_labels)

        return predictions, labels

    test_dataloader = get_test_dataloader(test_dataset)
    logits = get_logits(test_dataloader)
    predictions_df = group_predictions(test_dataset)
    y_pred, y_true = evaluate(predictions_df)
    results = classification_report(y_true, y_pred)

    print(results)

    with open(f"{output_dir}/results.txt", "w+") as f:
        f.write(results)
