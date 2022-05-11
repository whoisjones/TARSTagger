import random

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"tags"]):
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


def make_tars_dataset(dataset, tokenizer, tag2tars, tars_head, num_negatives):

    original_tags = dataset.features["ner_tags"].feature
    index2tag = {idx: tag for idx, tag in enumerate(original_tags.names)}

    tars2tag = {}
    for k, v in tag2tars.items():
        tars2tag[v] = tars2tag.get(v, []) + [k]

    def tars_labels(example):
        tars_labels = []
        for label in set(example["ner_tags"]):
            if index2tag.get(label) in tag2tars and index2tag.get(label) != "O":
                tars_labels.append(tag2tars.get(index2tag.get(label)))
        example["tars_labels"] = list(set(tars_labels))
        return example

    dataset = dataset.map(tars_labels)

    def tars_format(examples, num_negatives):

        all_tars_labels = set(tag2tars.values())
        all_tars_labels.remove("O")

        tars_formatted_tokens = []
        tars_formatted_tags = []

        for original_tokens, original_tags, tars_labels in zip(examples["tokens"], examples["ner_tags"], examples["tars_labels"]):

            original_bio_tags = [index2tag.get(x) for x in original_tags]
            original_tags_as_tars_labels = [tag2tars.get(index2tag.get(x)) for x in original_tags]

            tars_tags = []
            for original_bio_tag in original_bio_tags:
                for prefix in tars_head.keys():
                    if original_bio_tag.startswith(prefix):
                        tars_tags.append(prefix)

            filter_prefix = lambda x, y: x == y

            for positive_label in tars_labels:

                tars_label_prefix = positive_label.split() + [tokenizer.sep_token]
                tars_tokens = tars_label_prefix + original_tokens

                filtered_tars_tags = [tars_head.get("O")] * len(tars_label_prefix) + \
                                     [tars_head.get(tars_tag) if filter_prefix(positive_label, tars_prefix) else tars_head.get("O")
                                      for tars_tag, tars_prefix in zip(tars_tags, original_tags_as_tars_labels)]

                tars_formatted_tokens.append(tars_tokens)
                tars_formatted_tags.append(filtered_tars_tags)

            negative_samples = list(all_tars_labels.symmetric_difference(set(tars_labels)))
            if num_negatives > len(negative_samples):
                num_negatives = len(negative_samples)
            if len(negative_samples) > 0:
                negative_label = random.sample(negative_samples, num_negatives).pop()
                tars_tokens = negative_label.split() + [tokenizer.sep_token] + original_tokens
                filtered_tars_tags = [tars_head.get(tars_tag) for tars_tag in ["O"] * len(tars_tokens)]

                tars_formatted_tokens.append(tars_tokens)
                tars_formatted_tags.append(filtered_tars_tags)

        return {"tokens": tars_formatted_tokens, "tags": tars_formatted_tags}

    dataset = dataset.map(lambda p: tars_format(p, num_negatives=num_negatives), batched=True, remove_columns=dataset.column_names)

    dataset = dataset.map(lambda p: tokenize_and_align_labels(p, tokenizer), batched=True, remove_columns=["tokens", "tags"])

    return dataset
