from datasets import load_dataset

from .constants import *
from .processing import preprocess_corpus


def _load_corpus(dataset_name: str):
    if dataset_name in available_datasets:
        dataset = load_dataset(
            available_datasets.get(dataset_name).get(dataset_key),
            available_datasets.get(dataset_name).get(subset_key)
            if subset_key in available_datasets.get(dataset_name)
            else None,
        )
    else:
        raise ValueError(
            f"dataset {dataset_name} is not available. please choose from {[x for x in available_datasets.keys()]}"
        )

    return dataset


def load_corpus(dataset_name: str):
    dataset = _load_corpus(dataset_name)
    config = preprocess_corpus(
        dataset=dataset,
        dataset_name=dataset_name,
    )
    return config["dataset"], config["tags"], config["index2tag"], config["tag2index"]


def split_dataset(dataset):
    if "train" in dataset:
        train_dataset = dataset["train"]
    else:
        raise ValueError("train split does not exist in dataset.")
    if "validation" in dataset:
        val_dataset = dataset["validation"]
    else:
        raise ValueError("validation split does not exist in dataset.")
    if "test" in dataset:
        test_dataset = dataset["test"]
    else:
        raise ValueError("test split does not exist in dataset.")

    return train_dataset, val_dataset, test_dataset


def load_label_verbalizer_corpus():
    english_dataset, english_tags, english_idx2tag, english_tag2idx = load_corpus(
        "ontonotes"
    )
    arabic_dataset, arabic_tags, arabic_idx2tag, arabic_tag2idx = load_corpus("arabic")
    chinese_dataset, chinese_tags, chinese_idx2tag, chinese_tag2idx = load_corpus(
        "chinese"
    )

    english_target_tags = [english_tag2idx.get(tag) for tag in ["B-ORG", "I-ORG"]]
    arabic_target_tags = [arabic_tag2idx.get(tag) for tag in ["B-PERSON", "I-PERSON"]]
    chinese_target_tags = [chinese_tag2idx.get(tag) for tag in ["B-LOC", "I-LOC"]]
    all_target_tags = english_target_tags + arabic_target_tags + chinese_target_tags

    def transform_english(example):
        example["ner_tags"] = [
            x if x in english_target_tags else 0 for x in example["ner_tags"]
        ]
        return example

    def transform_arabic(example):
        example["ner_tags"] = [
            x if x in arabic_target_tags else 0 for x in example["ner_tags"]
        ]
        return example

    def transform_chinese(example):
        example["ner_tags"] = [
            x if x in chinese_target_tags else 0 for x in example["ner_tags"]
        ]
        return example

    def transform_test(example):
        example["ner_tags"] = [
            x if x in all_target_tags else 0 for x in example["ner_tags"]
        ]
        return example

    english_dataset["train"] = english_dataset["train"].map(transform_english)
    english_dataset["validation"] = english_dataset["validation"].map(transform_english)
    english_dataset["test"] = english_dataset["test"].map(transform_test)

    english_dataset["train"] = english_dataset["train"].filter(
        lambda example: any(i in example["ner_tags"] for i in english_target_tags)
    )
    english_dataset["validation"] = english_dataset["validation"].filter(
        lambda example: any(i in example["ner_tags"] for i in english_target_tags)
    )
    english_dataset["test"] = english_dataset["test"].filter(
        lambda example: any(i in example["ner_tags"] for i in english_target_tags)
    )

    arabic_dataset["train"] = arabic_dataset["train"].map(transform_arabic)
    arabic_dataset["validation"] = arabic_dataset["validation"].map(transform_arabic)
    arabic_dataset["test"] = arabic_dataset["test"].map(transform_test)

    arabic_dataset["train"] = arabic_dataset["train"].filter(
        lambda example: any(i in example["ner_tags"] for i in arabic_target_tags)
    )
    arabic_dataset["validation"] = arabic_dataset["validation"].filter(
        lambda example: any(i in example["ner_tags"] for i in arabic_target_tags)
    )
    arabic_dataset["test"] = arabic_dataset["test"].filter(
        lambda example: any(i in example["ner_tags"] for i in arabic_target_tags)
    )

    chinese_dataset["train"] = chinese_dataset["train"].map(transform_chinese)
    chinese_dataset["validation"] = chinese_dataset["validation"].map(transform_chinese)
    chinese_dataset["test"] = chinese_dataset["test"].map(transform_test)

    chinese_dataset["train"] = chinese_dataset["train"].filter(
        lambda example: any(i in example["ner_tags"] for i in chinese_target_tags)
    )
    chinese_dataset["validation"] = chinese_dataset["validation"].filter(
        lambda example: any(i in example["ner_tags"] for i in chinese_target_tags)
    )
    chinese_dataset["test"] = chinese_dataset["test"].filter(
        lambda example: any(i in example["ner_tags"] for i in chinese_target_tags)
    )

