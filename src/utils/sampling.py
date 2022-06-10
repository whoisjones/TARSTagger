import random

def k_shot_sampling(k, mapping, seed, mode):
    if mode == "soft":
        return k_shot_soft_sampling(k, mapping, seed)
    elif mode == "strict":
        return k_shot_strict_sampling(k, mapping, seed)
    else:
        raise ValueError(f"Unknown sampling strategy: {mode}.")

def k_shot_soft_sampling(k, mapping, seed):
    count = {label: 0 for label in mapping.keys()}
    total_examples = max([max(x) for x in mapping.values()])

    random.seed(seed)

    completed = False
    idx = 0
    k_shot_indices = []
    while not completed:
        sample = random.randint(0, total_examples)
        idx += 1

        if idx % total_examples == 0:
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
                if count[label] + 1 >= 2*k:
                    can_be_added = False

        if can_be_added:
            k_shot_indices.append(sample)
            for i in possible_indices:
                label = list(count.keys())[i]
                count[label] += 1
        else:
            continue

    return k_shot_indices


def k_shot_strict_sampling(k, mapping, seed):
    count = {label: 0 for label in mapping.keys()}
    total_examples = max([max(x) for x in mapping.values()])

    random.seed(seed)

    completed = False
    idx = 0
    k_shot_indices = []
    while not completed:
        sample = random.randint(0, total_examples)
        idx += 1

        if idx % total_examples == 0:
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
