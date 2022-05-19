from datasets import load_dataset

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


def load_corpus(dataset_name: str):
    dataset = _load_corpus(dataset_name)
    tags, index2tag, tag2index = _load_tag_mapping(dataset_name=dataset_name, dataset=dataset)
    return dataset, tags, index2tag, tag2index


def _load_corpus(dataset: str):
    dataset_key = "dataset_name"
    subset_key = "subset"

    available_datasets = {
        "conll": {dataset_key: "conll2003"},
        "spanish": {dataset_key: "conll2002", subset_key: "es"},
        "dutch": {dataset_key: "conll2002", subset_key: "nl"},
        "finnish": {dataset_key: "finer"},
        "ontonotes": {dataset_key: "conll2012_ontonotesv5", subset_key: "english_v12"},
        "arabic": {dataset_key: "conll2012_ontonotesv5", subset_key: "arabic_v4"},
        "chinese": {dataset_key: "conll2012_ontonotesv5", subset_key: "chinese_v4"}
    }

    if dataset in available_datasets:
        dataset = load_dataset(
            available_datasets.get(dataset).get(dataset_key),
            available_datasets.get(dataset).get(subset_key) if subset_key in available_datasets.get(dataset) else None,
        )
    else:
        raise ValueError(f"dataset {dataset} is not available. please choose from {[x for x in available_datasets.keys()]}")

    return dataset


def _load_tag_mapping(dataset_name: str, dataset):
    if dataset_name in ["conll", "spanish", "dutch", "finnish"]:
        tags = dataset["train"].features["ner_tags"].feature
    elif dataset_name in ["ontonotes", "arabic", "chinese"]:
        tags = dataset["train"].features["sentences"][0]["named_entities"].feature
    else:
        raise ValueError(f"dataset {dataset_name} unknown.")

    index2tag = {idx: tag for idx, tag in enumerate(tags.names)}
    tag2index = {tag: idx for idx, tag in enumerate(tags.names)}

    return tags, index2tag, tag2index


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
