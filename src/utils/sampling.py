import random

def k_shot_sampling(k, mapping, seed, mode):
    if mode == "soft":
        return k_shot_soft_sampling(k, mapping, seed)
    elif mode == "strict":
        return k_shot_strict_sampling(k, mapping, seed)
    else:
        raise ValueError(f"Unknown sampling strategy: {mode}.")

def k_shot_soft_sampling(k, mapping, seed):
    sorted_mapping = {key: val for key, val in sorted(mapping.items(), key=lambda item: len(item[1]))}
    count = {label: 0 for label in mapping.keys()}
    lower_bounds = {k: len(v) for k, v in mapping.items()}
    k_shot_indices = []

    random.seed(seed)

    completed = False
    while not completed:

        for label_key, sentence_ids in sorted_mapping.items():

            if completed:
                break

            if not k > len(sentence_ids):
                samples_for_label = random.sample(sentence_ids, k)
            else:
                samples_for_label = random.sample(sentence_ids, len(sentence_ids))

            for sentence_id in samples_for_label:
                labels_to_be_considered = [_key for _key, _vals in mapping.items() if sentence_id in _vals]
                if all([True if count[label] < k else False for label in labels_to_be_considered]) and sentence_id not in k_shot_indices:
                    k_shot_indices.append(sentence_id)
                    for label in labels_to_be_considered:
                        count[label] += 1

                    if all([c >= min(k, lb) for c, lb in zip(count.values(), lower_bounds.values())]):
                        completed = True
                        break

                elif count[label_key] < 2*k and any([True if count[label] < k else False for label in labels_to_be_considered]) and sentence_id not in k_shot_indices:
                    k_shot_indices.append(sentence_id)
                    for label in labels_to_be_considered:
                        count[label] += 1

                    if all([c >= min(k, lb) for c, lb in zip(count.values(), lower_bounds.values())]):
                        completed = True
                        break

    return k_shot_indices, count


def k_shot_strict_sampling(k, mapping, seed):
    count = {label: 0 for label in mapping.keys()}
    total_examples = list(set([x for s in mapping.values() for x in s]))

    random.seed(seed)

    completed = False
    idx = 0
    k_shot_indices = []
    while not completed:
        sample = random.sample(total_examples, 1)[0]
        idx += 1

        if idx % len(total_examples) == 0:
            k_shot_indices = []
            count = {label: 0 for label in mapping.keys()}

        if all([c == k for c in count.values()]):
            completed = True
            continue

        can_be_added = False
        sample_has_labels = [sample in mapping[label] for label in count.keys()]
        if any(sample_has_labels):
            can_be_added = True
            possible_indices = [i for i, x in enumerate(sample_has_labels) if x]
            for i in possible_indices:
                label = list(count.keys())[i]
                if count[label] + 1 > k:
                    can_be_added = False

        if can_be_added:
            k_shot_indices.append(sample)
            for i in possible_indices:
                label = list(count.keys())[i]
                count[label] += 1
        else:
            continue

    return k_shot_indices
