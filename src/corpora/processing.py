import datasets
from .constants import *


def preprocess_corpus(**config):
    """
    Corpora specific processing.
    :param config: key-value pairs consisting out of dataset, dataset_name, tags, index2tag and tag2index
    :return: processed config key-value pair
    """
    if config["dataset_name"] in all_ontonotes:
        config = _preprocess_ontonotes(**config)

    if config["dataset_name"] in tagset_extension_datasets:
        config = _preprocess_tagset_extension(**config)

    if config["dataset_name"] in fewnerd_datasets:
        config = _preprocess_fewnerd(**config)

    if config["dataset_name"] == "arabic":
        config = _preprocess_arabic(**config)

    if config["dataset_name"] in ["finnish", "conll"]:
        config = _preprocess_id_to_int(**config)

    if not config["dataset_name"] in tagset_extension_datasets:
        config["tags"] = config["dataset"]["train"].features["ner_tags"].feature
        config["index2tag"] = {idx: tag for idx, tag in enumerate(config["tags"].names)}
        config["tag2index"] = {tag: idx for idx, tag in enumerate(config["tags"].names)}

    return config


def _convert_ontonotes_format(examples):
    tokens = []
    labels = []
    for sentences in examples["sentences"]:
        for sentence in sentences:
            tokens.append(sentence["words"])
            labels.append(sentence["named_entities"])
    return {"tokens": tokens, "ner_tags": labels}


def _preprocess_ontonotes(**config):
    """
    Removes document abstraction and returns only sentences for datasets plus renaming of named entities column.
    :param config: key-value pairs consisting out of dataset, dataset_name, tags, index2tag and tag2index
    :return: processed config key-value pair
    """
    features = {
        split: config["dataset"]["train"].features["sentences"][0]["named_entities"]
        for split in config["dataset"]
    }
    config["dataset"] = config["dataset"].map(
        _convert_ontonotes_format,
        batched=True,
        remove_columns=config["dataset"]["train"].column_names,
    )
    for split, feature in features.items():
        config["dataset"][split].features["ner_tags"] = feature
        config["dataset"][split] = datasets.concatenate_datasets(
            [
                datasets.Dataset.from_dict(
                    {"id": range(0, len(config["dataset"][split]))}
                ),
                config["dataset"][split],
            ],
            axis=1,
            split=config["dataset"][split].split,
        )

    return config


def _preprocess_tagset_extension(**config):
    """
    Masks target tags as 0 in order to perform tag set extension experiments.
    :param config: key-value pairs consisting out of dataset, dataset_name, tags, index2tag and tag2index
    :return: processed config key-value pair
    """
    import copy
    _tags_template = copy.deepcopy(config["dataset"]["train"].features["ner_tags"].feature)
    _full_tags = config["dataset"]["train"].features["ner_tags"].feature
    _full_index2tag = {idx: tag for idx, tag in enumerate(_tags_template.names)}
    _full_tag2index = {tag: idx for idx, tag in enumerate(_tags_template.names)}

    if config["dataset_name"] == "ontonotes_AB":
        _tags_template.names = list(set(group_a + group_b))
        _tags_template.num_classes = len(list(set(group_a + group_b)))
    elif config["dataset_name"] == "ontonotes_BC":
        _tags_template.names = list(set(group_b + group_c))
        _tags_template.num_classes = len(list(set(group_b + group_c)))
    elif config["dataset_name"] == "ontonotes_AC":
        _tags_template.names = list(set(group_a + group_c))
        _tags_template.num_classes = len(list(set(group_a + group_c)))
    elif config["dataset_name"] == "ontonotes_A":
        _tags_template.names = list(set(group_a))
        _tags_template.num_classes = len(group_a)
    elif config["dataset_name"] == "ontonotes_B":
        _tags_template.names = list(set(group_b))
        _tags_template.num_classes = len(list(set(group_b)))
    elif config["dataset_name"] == "ontonotes_C":
        _tags_template.names = list(set(group_c))
        _tags_template.num_classes = len(list(set(group_c)))
    elif config["dataset_name"] in ["ontonotes_15", "chinese_15", "arabic_15"]:
        _labels = [tag for tag in _full_tags.names if tag not in cross_lingual_eval_tags]
        _labels.append("O")
        _tags_template.names = list(_labels)
        _tags_template.num_classes = len(_labels)
    elif config["dataset_name"] in ["ontonotes_3", "chinese_3", "arabic_3"]:
        _tags_template.names = cross_lingual_eval_tags
        _tags_template.num_classes = len(cross_lingual_eval_tags)

    config["tags"], config["dataset"]["train"].features["ner_tags"].feature = _tags_template, _tags_template

    config["index2tag"] = {k: v for k, v in _full_index2tag.items() if v in config["tags"].names}
    config["tag2index"] = {v: k for k, v in _full_index2tag.items() if v in config["tags"].names}

    config["dataset"]["train"] = config["dataset"]["train"].filter(lambda example: all([elem in config["tags"].names for elem in [_full_index2tag.get(x) for x in example["ner_tags"]]]))
    config["dataset"]["validation"] = config["dataset"]["validation"].filter(lambda example: all([elem in config["tags"].names for elem in [_full_index2tag.get(x) for x in example["ner_tags"]]]))

    def mask_labels(example):
        example["ner_tags"] = [x if x in config["tags"].names else _full_tag2index.get("O") for x in example["ner_tags"]]
        return example

    config["dataset"]["test"] = config["dataset"]["test"].map(mask_labels)

    return config


def _convert_fewnerd_format(examples):
    return {
        "id": examples["id"],
        "tokens": examples["tokens"],
        "ner_tags": examples["fine_ner_tags"],
    }


def _preprocess_fewnerd(**config):
    """
    Rename fewnerd named entities column.
    :param config: key-value pairs consisting out of dataset, dataset_name, tags, index2tag and tag2index
    :return: processed config key-value pair
    """
    features = {
        split: config["dataset"]["train"].features["fine_ner_tags"]
        for split in config["dataset"]
    }
    config["dataset"] = config["dataset"].map(
        _convert_fewnerd_format,
        batched=True,
        remove_columns=config["dataset"]["train"].column_names,
    )
    for split, feature in features.items():
        config["dataset"][split].features["ner_tags"] = feature

    return config


def _convert_arabic(example):
    example["tokens"] = [token.split("#").pop(0) for token in example["tokens"]]
    return example


def _preprocess_arabic(**config):
    """
    Remove linguistic annotations in front of sentences
    :param config: key-value pairs consisting out of dataset, dataset_name, tags, index2tag and tag2index
    :return: processed config key-value pair
    """
    config["dataset"] = config["dataset"].map(_convert_arabic)
    return config


def _convert_id_to_int(example):
    example["id"] = int(example["id"])
    return example


def _preprocess_id_to_int(**config):
    """
    Convert ID to int if it is string.
    :param config: key-value pairs consisting out of dataset, dataset_name, tags, index2tag and tag2index
    :return: processed config key-value pair
    """
    config["dataset"] = config["dataset"].map(_convert_id_to_int)
    return config
