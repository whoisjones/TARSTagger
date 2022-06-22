import random

from .tokenization import tokenize_and_align_tars_labels
from ...corpora.constants import label_name_map, chinese_name_map, arabic_name_map, finnish_name_map, alphabetical_label_name_map, unprocessed_label_name_map


def make_tars_datasets(
    datasets: list, tokenizer, index2tag, org_tag2tars_label, tars_tag2id, num_negatives
):
    """
    transform a dataset into TARS format.
    :param datasets:
    :param tokenizer:
    :param index2tag:
    :param org_tag2tars_label:
    :param tars_tag2id:
    :return:
    """
    if not len(datasets) == 3:
        raise ValueError(
            "datasets attribute should be a list of datasets (train, val, test)."
        )

    tars_datasets = []
    for dataset in datasets:
        processed_dataset = _make_tars_dataset(
            dataset=dataset,
            tokenizer=tokenizer,
            index2tag=index2tag,
            org_tag2tars_label=org_tag2tars_label,
            tars_tag2id=tars_tag2id,
            num_negatives="all" if dataset.split._name == "test" else num_negatives,
        )
        tars_datasets.append(processed_dataset)

    return tuple(tars_datasets)


def _make_tars_dataset(
    dataset,
    tokenizer,
    index2tag,
    org_tag2tars_label,
    tars_tag2id,
    num_negatives: str = "one",
):
    def tars_labels(example):
        tars_labels = []
        for label in set(example["ner_tags"]):
            if (
                index2tag.get(label) in org_tag2tars_label
                and index2tag.get(label) != "O"
            ):
                tars_labels.append(org_tag2tars_label.get(index2tag.get(label)))
        example["tars_labels"] = list(set(tars_labels))
        return example

    dataset = dataset.map(tars_labels)

    def tars_format(examples, num_negatives):

        all_tars_labels = set(org_tag2tars_label.values())
        if "O" in all_tars_labels:
            all_tars_labels.remove("O")

        output_tars_formatted_tokens = []
        output_tars_formatted_tags = []
        output_tars_label = []
        output_original_tags = []
        output_ids = []
        output_label_lengths = []

        for idx, original_tokens, original_tags, tars_labels in zip(
            examples["id"],
            examples["tokens"],
            examples["ner_tags"],
            examples["tars_labels"],
        ):

            original_bio_tags = [index2tag.get(x) for x in original_tags]
            original_tags_as_tars_labels = [
                org_tag2tars_label.get(index2tag.get(x)) for x in original_tags
            ]

            tars_tags = []
            for original_bio_tag in original_bio_tags:
                for prefix in tars_tag2id.keys():
                    if original_bio_tag.startswith(prefix):
                        tars_tags.append(prefix)

            filter_prefix = lambda x, y: x == y

            for positive_label in tars_labels:

                tars_label_prefix = positive_label.split() + [tokenizer.sep_token]
                tars_tokens = tars_label_prefix + original_tokens

                filtered_tars_tags = [tars_tag2id.get("O")] * len(tars_label_prefix) + [
                    tars_tag2id.get(tars_tag)
                    if filter_prefix(positive_label, tars_prefix)
                    else tars_tag2id.get("O")
                    for tars_tag, tars_prefix in zip(
                        tars_tags, original_tags_as_tars_labels
                    )
                ]

                output_ids.append(idx)
                output_original_tags.append(original_tags)
                output_tars_formatted_tokens.append(tars_tokens)
                output_tars_formatted_tags.append(filtered_tars_tags)
                output_tars_label.append(positive_label)
                output_label_lengths.append(len(tars_label_prefix))

            if not num_negatives == "none":
                negative_samples = list(
                    all_tars_labels.symmetric_difference(set(tars_labels))
                )
                if len(negative_samples) > 0 and num_negatives in ["one", "two"]:
                    if num_negatives == "two" and len(negative_samples) > 1:
                        negative_labels = random.sample(negative_samples, 2)
                    else:
                        negative_labels = random.sample(negative_samples, 1)

                    for negative_label in negative_labels:
                        tars_label_prefix = negative_label.split() + [tokenizer.sep_token]
                        tars_tokens = tars_label_prefix + original_tokens
                        filtered_tars_tags = [
                            tars_tag2id.get(tars_tag) for tars_tag in ["O"] * len(tars_tokens)
                        ]

                        output_ids.append(idx)
                        output_original_tags.append(original_tags)
                        output_tars_formatted_tokens.append(tars_tokens)
                        output_tars_formatted_tags.append(filtered_tars_tags)
                        output_tars_label.append(negative_label)
                        output_label_lengths.append(len(tars_label_prefix))

                elif len(negative_samples) > 0 and num_negatives == "all":
                    for negative_label in negative_samples:
                        tars_label_prefix = negative_label.split() + [tokenizer.sep_token]
                        tars_tokens = tars_label_prefix + original_tokens
                        filtered_tars_tags = [
                            tars_tag2id.get(tars_tag)
                            for tars_tag in ["O"] * len(tars_tokens)
                        ]

                        output_ids.append(idx)
                        output_original_tags.append(original_tags)
                        output_tars_formatted_tokens.append(tars_tokens)
                        output_tars_formatted_tags.append(filtered_tars_tags)
                        output_tars_label.append(negative_label)
                        output_label_lengths.append(len(tars_label_prefix))

        return {
            "id": output_ids,
            "tokens": output_tars_formatted_tokens,
            "tars_tags": output_tars_formatted_tags,
            "ner_tags": output_original_tags,
            "tars_label_length": output_label_lengths,
            "tars_labels": output_tars_label,
        }

    dataset = dataset.map(
        lambda p: tars_format(p, num_negatives),
        batched=True,
        remove_columns=dataset.column_names,
    )

    dataset = dataset.map(
        lambda p: tokenize_and_align_tars_labels(p, tokenizer),
        batched=True,
        remove_columns=["tokens"],
    )

    return dataset


def load_tars_label_mapping(tags):
    org_tag2tars_label = {tag: label_name_map.get(tag) for tag in tags.names}

    tars_label2org_tag = {}
    for k, v in org_tag2tars_label.items():
        tars_label2org_tag[v] = tars_label2org_tag.get(v, []) + [k]

    return org_tag2tars_label, tars_label2org_tag

def load_tars_alphabetical_label_mapping(tags):
    _label_name_map = alphabetical_label_name_map

    org_tag2tars_label = {tag: _label_name_map.get(tag) for tag in tags.names}

    tars_label2org_tag = {}
    for k, v in org_tag2tars_label.items():
        tars_label2org_tag[v] = tars_label2org_tag.get(v, []) + [k]

    return org_tag2tars_label, tars_label2org_tag

def load_tars_unprocessed_label_mapping(tags):
    _label_name_map = unprocessed_label_name_map

    org_tag2tars_label = {tag: _label_name_map.get(tag) for tag in tags.names}

    tars_label2org_tag = {}
    for k, v in org_tag2tars_label.items():
        tars_label2org_tag[v] = tars_label2org_tag.get(v, []) + [k]

    return org_tag2tars_label, tars_label2org_tag

def load_tars_cross_lingual_label_mapping(tags, corpus):
    if corpus.startswith("ontonotes"):
        _label_name_map = {k: v for k, v in label_name_map.items() if k in chinese_name_map.keys()}
    elif corpus.startswith("chinese"):
        _label_name_map = chinese_name_map
    elif corpus.startswith("arabic"):
        _label_name_map = arabic_name_map
    elif corpus.startswith("finnish"):
        _label_name_map = finnish_name_map
    else:
        raise ValueError("unknown language.")
    org_tag2tars_label = {tag: _label_name_map.get(tag) for tag in tags.names}

    tars_label2org_tag = {}
    for k, v in org_tag2tars_label.items():
        tars_label2org_tag[v] = tars_label2org_tag.get(v, []) + [k]

    return org_tag2tars_label, tars_label2org_tag
