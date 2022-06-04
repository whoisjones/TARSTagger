import random
from typing import List, Any


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def convert_ontonotes_format(examples):
    tokens = []
    labels = []
    for sentences in examples["sentences"]:
        for sentence in sentences:
            tokens.append(sentence["words"])
            labels.append(sentence["named_entities"])
    return {"tokens": tokens, "ner_tags": labels}


def convert_fewnerd_format(examples):
    return {"id": examples["id"], "tokens": examples["tokens"], "ner_tags": examples["fine_ner_tags"]}


def split_arabic(example):
    example["tokens"] = [token.split("#").pop(0) for token in example["tokens"]]
    return example

def convert_finnish_format(example):
    example["id"] = int(example["id"])
    return example

def tokenize_and_align_tars_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, (label, label_length) in enumerate(zip(examples["tars_tags"], examples["tars_label_length"])):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx and word_idx >= label_length:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def make_tars_datasets(datasets: list, tokenizer, index2tag, org_tag2tars_label, tars_tag2id):
    if not len(datasets) == 3:
        raise ValueError("datasets attribute should be a list of datasets (train, val, test).")

    tars_datasets = []
    for dataset in datasets:
        processed_dataset = make_tars_dataset(dataset=dataset,
                                              tokenizer=tokenizer,
                                              index2tag=index2tag,
                                              org_tag2tars_label=org_tag2tars_label,
                                              tars_tag2id=tars_tag2id,
                                              num_negatives="all" if dataset.split._name == "test" else "one")
        tars_datasets.append(processed_dataset)

    return tuple(tars_datasets)


def make_tars_dataset(dataset, tokenizer, index2tag, org_tag2tars_label, tars_tag2id, num_negatives: str = "one"):

    def tars_labels(example):
        tars_labels = []
        for label in set(example["ner_tags"]):
            if index2tag.get(label) in org_tag2tars_label and index2tag.get(label) != "O":
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

        for idx, original_tokens, original_tags, tars_labels in zip(examples["id"], examples["tokens"], examples["ner_tags"], examples["tars_labels"]):

            original_bio_tags = [index2tag.get(x) for x in original_tags]
            original_tags_as_tars_labels = [org_tag2tars_label.get(index2tag.get(x)) for x in original_tags]

            tars_tags = []
            for original_bio_tag in original_bio_tags:
                for prefix in tars_tag2id.keys():
                    if original_bio_tag.startswith(prefix):
                        tars_tags.append(prefix)

            filter_prefix = lambda x, y: x == y

            for positive_label in tars_labels:

                tars_label_prefix = positive_label.split() + [tokenizer.sep_token]
                tars_tokens = tars_label_prefix + original_tokens

                filtered_tars_tags = [tars_tag2id.get("O")] * len(tars_label_prefix) + \
                                     [tars_tag2id.get(tars_tag) if filter_prefix(positive_label, tars_prefix) else tars_tag2id.get("O")
                                      for tars_tag, tars_prefix in zip(tars_tags, original_tags_as_tars_labels)]

                output_ids.append(idx)
                output_original_tags.append(original_tags)
                output_tars_formatted_tokens.append(tars_tokens)
                output_tars_formatted_tags.append(filtered_tars_tags)
                output_tars_label.append(positive_label)
                output_label_lengths.append(len(tars_label_prefix))

            negative_samples = list(all_tars_labels.symmetric_difference(set(tars_labels)))
            if len(negative_samples) > 0 and num_negatives == "one":
                negative_label = random.sample(negative_samples, 1).pop()
                tars_label_prefix = negative_label.split() + [tokenizer.sep_token]
                tars_tokens = tars_label_prefix + original_tokens
                filtered_tars_tags = [tars_tag2id.get(tars_tag) for tars_tag in ["O"] * len(tars_tokens)]

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
                    filtered_tars_tags = [tars_tag2id.get(tars_tag) for tars_tag in ["O"] * len(tars_tokens)]

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
            "tars_labels": output_tars_label
        }

    dataset = dataset.map(lambda p: tars_format(p, num_negatives), batched=True, remove_columns=dataset.column_names)

    dataset = dataset.map(lambda p: tokenize_and_align_tars_labels(p, tokenizer), batched=True, remove_columns=["tokens"])

    return dataset

def k_shot_sampling(k, mapping, seed):
    count = {label: 0 for label in mapping.keys()}
    total_examples = max([max(x) for x in mapping.values()])

    random.seed(seed)

    completed = False
    while not completed:
        k_shot_indices = []
        indices = list(range(0, total_examples))
        random.shuffle(indices)

        for sample in indices:
            add = []
            for label in count.keys():
                if sample in mapping[label]:
                    if count[label] + 1 < 2 * k:
                        count[label] += 1
                        add.append(True)
                    else:
                        add.append(False)
                else:
                    add.append(False)

            if any(add):
                k_shot_indices.append(sample)

                k_complete = []
                for label in mapping.keys():
                    if count[label] == len(mapping[label]) or count[label] >= k:
                        k_complete.append(True)
                    else:
                        k_complete.append(False)

                if all(k_complete):
                    completed = True
                    break

    return k_shot_indices
