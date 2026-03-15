import sys
from typing import List, Tuple, Dict


def read_input_tokens() -> List[str]:
    data = sys.stdin.read()
    return data.strip().split()


def parse_int(token: str) -> int:
    # Inputs may provide integers as float-like strings (e.g., "1.0").
    try:
        return int(token)
    except ValueError:
        return int(float(token))


def knn_classify(
    k: int,
    num_samples: int,
    num_dims: int,
    _num_classes: int,
    query: List[float],
    samples: List[Tuple[List[float], int]],
) -> Tuple[int, int]:
    # Compute squared Euclidean distances to avoid unnecessary sqrt
    distances: List[Tuple[float, int]] = []
    for features, label in samples:
        sq_dist = 0.0
        for i in range(num_dims):
            diff = features[i] - query[i]
            sq_dist += diff * diff
        distances.append((sq_dist, label))

    distances.sort(key=lambda x: x[0])
    topk = distances[:k]

    label_counts: Dict[int, int] = {}
    for _, label in topk:
        label_counts[label] = label_counts.get(label, 0) + 1

    # Majority voting
    max_count = max(label_counts.values())
    candidates = {label for label, count in label_counts.items() if count == max_count}

    if len(candidates) == 1:
        winner = next(iter(candidates))
        return winner, max_count

    # Tie-break: among tied labels, choose the one whose nearest neighbor is closest
    for dist, label in topk:
        if label in candidates:
            winner = label
            return winner, label_counts[winner]

    # Fallback (should not reach here)
    any_label = next(iter(label_counts))
    return any_label, label_counts[any_label]


def main() -> None:
    tokens = read_input_tokens()
    it = iter(tokens)

    try:
        k = parse_int(next(it))
        n = parse_int(next(it))
        d = parse_int(next(it))
        c = parse_int(next(it))
    except StopIteration:
        return

    query: List[float] = []
    for _ in range(d):
        try:
            query.append(float(next(it)))
        except StopIteration:
            return

    samples: List[Tuple[List[float], int]] = []
    for _ in range(n):
        features: List[float] = []
        for _ in range(d):
            try:
                features.append(float(next(it)))
            except StopIteration:
                return
        try:
            raw_label = next(it)
        except StopIteration:
            return
        label = parse_int(raw_label)
        samples.append((features, label))

    label, count = knn_classify(k, n, d, c, query, samples)
    # Output as integers: "label count"
    sys.stdout.write(f"{label} {count}")


if __name__ == "__main__":
    main()

