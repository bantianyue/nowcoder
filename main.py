import sys
from collections import Counter


def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return

    it = iter(data)
    try:
        k = int(next(it))
        num_samples = int(next(it))
        num_dims = int(next(it))
        _num_classes = int(next(it))  # not directly needed for computation
    except StopIteration:
        return

    # Query point
    try:
        query = [float(next(it)) for _ in range(num_dims)]
    except StopIteration:
        return

    samples = []  # list[tuple[list[float], int]]
    for _ in range(num_samples):
        try:
            features = [float(next(it)) for _ in range(num_dims)]
            label_val = float(next(it))
        except StopIteration:
            return
        # Labels are given like 0.0, 1.0; cast to int safely
        label = int(round(label_val))
        samples.append((features, label))

    # Compute squared Euclidean distances (monotonic to Euclidean)
    distances = []  # list[tuple[float, int]] (distance, label)
    for features, label in samples:
        dist_sq = 0.0
        for a, b in zip(query, features):
            d = a - b
            dist_sq += d * d
        distances.append((dist_sq, label))

    # Sort by distance ascending; tie-break by label for determinism
    distances.sort(key=lambda x: (x[0], x[1]))

    topk = distances[:k]
    label_counts = Counter(label for _, label in topk)
    max_count = max(label_counts.values())
    majority_labels = [lbl for lbl, cnt in label_counts.items() if cnt == max_count]

    if len(majority_labels) == 1:
        chosen_label = majority_labels[0]
    else:
        majority_set = set(majority_labels)
        # Choose the nearest neighbor among the tied majority labels
        chosen_label = None
        for dist, lbl in topk:
            if lbl in majority_set:
                chosen_label = lbl
                break
        if chosen_label is None:
            # Fallback (should not happen): choose smallest label among majority
            chosen_label = min(majority_set)

    print(f"{chosen_label} {label_counts[chosen_label]}")


if __name__ == "__main__":
    main()

