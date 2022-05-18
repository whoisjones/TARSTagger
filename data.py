import random

def tokenize_and_align_labels(examples, tokenizer):
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


def make_tars_dataset(dataset, tokenizer, index2tag, tag2tars, tars_head, num_negatives: str = "one"):

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
                filtered_tars_tags = [tars_head.get(tars_tag) for tars_tag in ["O"] * len(tars_tokens)]

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
                    filtered_tars_tags = [tars_head.get(tars_tag) for tars_tag in ["O"] * len(tars_tokens)]

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

    dataset = dataset.map(lambda p: tokenize_and_align_labels(p, tokenizer), batched=True, remove_columns=["tokens"])

    return dataset
