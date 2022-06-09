from .constants import label_name_map


def load_label_id_mapping(dataset, tags, index2tag):
    label2id = {idx: label_name_map.get(tag) for idx, tag in enumerate(tags.names)}
    label2id = {v: [] for v in label2id.values()}

    for example in dataset:
        for tag in set(
            [
                label_name_map.get(index2tag.get(_tag))
                for _tag in example["ner_tags"]
            ]
        ):
            label2id[tag].append(example["id"])

    if "O" in label2id:
        del label2id["O"]

    label2id = {k: v for k, v in label2id.items() if len(v) > 0}

    return label2id