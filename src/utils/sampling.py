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

def k_shot_strict_sampling(k, mapping, seed):
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
                    if count[label] + 1 <= k:
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
