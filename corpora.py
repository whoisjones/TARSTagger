import datasets
from datasets import load_dataset
from preprocessing import convert_ontonotes_format, convert_fewnerd_format, split_arabic, convert_finnish_format

tars_label_name_map = {"O": "O",
                       "B-PER": "person", "I-PER": "person",
                       "B-ORG": "organization", "I-ORG": "organization",
                       "B-LOC": "location", "I-LOC": "location",
                       "B-MISC": "miscellaneous", "I-MISC": "miscellaneous",
                       "B-PERSON": "person", "I-PERSON": "person",
                       "B-FAC": "building", "I-FAC": "building",
                       "B-CARDINAL": "cardinal", "I-CARDINAL": "cardinal",
                       "B-EVENT": "event", "I-EVENT": "event",
                       "B-GPE": "geopolitical", "I-GPE": "geopolitical",
                       "B-LANGUAGE": "language", "I-LANGUAGE": "language",
                       "B-LAW": "law", "I-LAW": "law",
                       "B-MONEY": "money", "I-MONEY": "money",
                       "B-NORP": "affiliation", "I-NORP": "affiliation",
                       "B-ORDINAL": "ordinal", "I-ORDINAL": "ordinal",
                       "B-PERCENT": "percentage", "I-PERCENT": "percentage",
                       "B-PRODUCT": "product", "I-PRODUCT": "product",
                       "B-QUANTITY": "quantity", "I-QUANTITY": "quantity",
                       "B-TIME": "time", "I-TIME": "time",
                       "B-WORK_OF_ART": "art", "I-WORK_OF_ART": "art",
                       "B-PRO": "product", "I-PRO": "product",
                       "B-DATE": "date", "I-DATE": "date",
                       }

standard_datasets = ["conll", "spanish", "dutch", "finnish"]
ontonotes_datasets = ["ontonotes", "arabic", "chinese"]
fewnerd_datasets = ["fewnerd-inter", "fewnerd-intra", "fewnerd-supervised"]
tagset_extension_datasets = ["ontonotes_AB", "ontonotes_AC", "ontonotes_BC",
                             "ontonotes_A", "ontonotes_B", "ontonotes_C"]

group_a = ["B-ORG", "B-NORP", "B-ORDINAL", "B-WORK_OF_ART", "B-QUANTITY", "B-LAW",
           "I-ORG", "I-NORP", "I-ORDINAL", "I-WORK_OF_ART", "I-QUANTITY", "I-LAW"]

group_b = ["B-GPE", "B-CARDINAL", "B-PERCENT", "B-TIME", "B-EVENT", "B-LANGUAGE",
           "I-GPE", "I-CARDINAL", "I-PERCENT", "I-TIME", "I-EVENT", "I-LANGUAGE"]

group_c = ["B-PERSON", "B-DATE", "B-MONEY", "B-LOC", "B-FAC", "B-PRODUCT",
           "I-PERSON", "I-DATE", "I-MONEY", "I-LOC", "I-FAC", "I-PRODUCT"]

def load_corpus(dataset_name: str):
    dataset = _load_corpus(dataset_name)
    tags, index2tag, tag2index = _load_tag_mapping(dataset)

    return dataset, tags, index2tag, tag2index


def _load_corpus(dataset_name: str):
    dataset_key = "dataset_name"
    subset_key = "subset"

    available_datasets = {
        "conll": {dataset_key: "conll2003"},
        "spanish": {dataset_key: "conll2002", subset_key: "es"},
        "dutch": {dataset_key: "conll2002", subset_key: "nl"},
        "finnish": {dataset_key: "finer"},
        "ontonotes": {dataset_key: "conll2012_ontonotesv5", subset_key: "english_v12"},
        "arabic": {dataset_key: "conll2012_ontonotesv5", subset_key: "arabic_v4"},
        "chinese": {dataset_key: "conll2012_ontonotesv5", subset_key: "chinese_v4"},
        "fewnerd-inter": {dataset_key: "dfki-nlp/few-nerd", subset_key: "inter"},
        "fewnerd-intra": {dataset_key: "dfki-nlp/few-nerd", subset_key: "intra"},
        "fewnerd-supervised": {dataset_key: "dfki-nlp/few-nerd", subset_key: "supervised"},
        "ontonotes_AB": {dataset_key: "conll2012_ontonotesv5", subset_key: "english_v12"},
        "ontonotes_AC": {dataset_key: "conll2012_ontonotesv5", subset_key: "english_v12"},
        "ontonotes_BC": {dataset_key: "conll2012_ontonotesv5", subset_key: "english_v12"},
        "ontonotes_A": {dataset_key: "conll2012_ontonotesv5", subset_key: "english_v12"},
        "ontonotes_B": {dataset_key: "conll2012_ontonotesv5", subset_key: "english_v12"},
        "ontonotes_C": {dataset_key: "conll2012_ontonotesv5", subset_key: "english_v12"},
    }

    if dataset_name in available_datasets:
        dataset = load_dataset(
            available_datasets.get(dataset_name).get(dataset_key),
            available_datasets.get(dataset_name).get(subset_key) if subset_key in available_datasets.get(dataset_name) else None,
        )
    else:
        raise ValueError(f"dataset {dataset_name} is not available. please choose from {[x for x in available_datasets.keys()]}")

    if dataset_name in ontonotes_datasets or dataset_name in tagset_extension_datasets:
        features = {split: dataset["train"].features["sentences"][0]["named_entities"] for split in dataset}
        dataset = dataset.map(convert_ontonotes_format, batched=True, remove_columns=dataset["train"].column_names)
        for split, feature in features.items():
            dataset[split].features["ner_tags"] = feature
            dataset[split] = datasets.concatenate_datasets([
                datasets.Dataset.from_dict({"id": range(0, len(dataset[split]))}),
                dataset[split]
            ], axis=1, split=dataset[split].split)

    if dataset_name in tagset_extension_datasets:
        tags = dataset["train"].features["ner_tags"].feature
        index2tag = {idx: tag for idx, tag in enumerate(tags.names)}

        if dataset_name == "ontonotes_AB":
            tags = group_a + group_b
        elif dataset_name == "ontonotes_BC":
            tags = group_b + group_c
        elif dataset_name == "ontonotes_AC":
            tags = group_a + group_c
        elif dataset_name == "ontonotes_A":
            tags = group_a
        elif dataset_name == "ontonotes_B":
            tags = group_b
        elif dataset_name == "ontonotes_C":
            tags = group_c

        def tag_transform(example):
            example["ner_tags"] = [x if index2tag[x] in tags else 0 for x in example["ner_tags"]]
            return example

        dataset["train"] = dataset["train"].map(tag_transform)
        dataset["validation"] = dataset["validation"].map(tag_transform)
        dataset["test"] = dataset["test"].map(tag_transform)

    if dataset_name in fewnerd_datasets:
        features = {split: dataset["train"].features["fine_ner_tags"] for split in dataset}
        dataset = dataset.map(convert_fewnerd_format, batched=True, remove_columns=dataset["train"].column_names)
        for split, feature in features.items():
            dataset[split].features["ner_tags"] = feature

    if dataset_name == "arabic":
        dataset = dataset.map(split_arabic)

    if dataset_name == "finnish":
        dataset = dataset.map(convert_finnish_format)

    return dataset


def _load_tag_mapping(dataset):
    tags = dataset["train"].features["ner_tags"].feature
    index2tag = {idx: tag for idx, tag in enumerate(tags.names)}
    tag2index = {tag: idx for idx, tag in enumerate(tags.names)}

    return tags, index2tag, tag2index

def load_label_id_mapping(dataset):
    tags = dataset.features["ner_tags"].feature
    index2tag = {idx: tag for idx, tag in enumerate(tags.names)}
    label2id = {idx: tars_label_name_map.get(tag) for idx, tag in enumerate(tags.names)}
    label2id = {v: [] for v in label2id.values()}

    for example in dataset:
        for tag in set([tars_label_name_map.get(index2tag.get(_tag)) for _tag in example["ner_tags"]]):
            label2id[tag].append(example["id"])

    if "O" in label2id:
        del label2id["O"]

    label2id = {k: v for k, v in label2id.items() if len(v) > 0}

    return label2id

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


def load_tars_mapping(tags):
    org_tag2tars_label = {tag: tars_label_name_map.get(tag) for tag in tags.names}

    tars_label2org_tag = {}
    for k, v in org_tag2tars_label.items():
        tars_label2org_tag[v] = tars_label2org_tag.get(v, []) + [k]

    return org_tag2tars_label, tars_label2org_tag
