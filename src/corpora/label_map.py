def load_label_id_mapping(dataset, index2tag):
    label2id = {v: [] for k, v in index2tag.items()}

    for example in dataset:
        for tag in set(
            [
                index2tag.get(_tag) for _tag in example["ner_tags"]
            ]
        ):
            label2id[tag].append(example["id"])

    if "O" in label2id:
        del label2id["O"]

    label2id = {k: v for k, v in label2id.items() if len(v) > 0}

    return label2id
