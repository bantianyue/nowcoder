import sys
from typing import Dict, Tuple, List


def parse_input_from_stdin() -> Tuple[str, Dict[str, int], Dict[Tuple[str, str], int]]:
    raw = sys.stdin.read()
    if not raw:
        return "", {}, {}

    # Be tolerant to extra blank lines and spaces
    lines = [line.strip() for line in raw.splitlines() if line.strip() != ""]
    it = iter(lines)

    try:
        text = next(it)
    except StopIteration:
        return "", {}, {}

    try:
        n = int(next(it))
    except StopIteration:
        return text, {}, {}

    vocab_scores: Dict[str, int] = {}
    for _ in range(n):
        try:
            line = next(it)
        except StopIteration:
            break
        parts = line.split()
        if len(parts) < 2:
            continue
        token, score_str = parts[0], parts[1]
        try:
            vocab_scores[token] = int(score_str)
        except ValueError:
            # Ignore malformed entries
            continue

    try:
        m = int(next(it))
    except StopIteration:
        return text, vocab_scores, {}
    except ValueError:
        m = 0

    transitions: Dict[Tuple[str, str], int] = {}
    for _ in range(m):
        try:
            line = next(it)
        except StopIteration:
            break
        parts = line.split()
        if len(parts) < 3:
            continue
        a, b, x_str = parts[0], parts[1], parts[2]
        try:
            transitions[(a, b)] = int(x_str)
        except ValueError:
            # Ignore malformed entries
            continue

    return text, vocab_scores, transitions


def compute_best_segmentation_score(
    text: str,
    vocab_scores: Dict[str, int],
    transitions: Dict[Tuple[str, str], int],
) -> int:
    if not text:
        return 0
    if not vocab_scores:
        return 0

    text_length: int = len(text)
    token_length_set = {len(token) for token in vocab_scores.keys()}
    if not token_length_set:
        return 0

    # dp[i]: mapping from the last token used to the best score for text[:i]
    dp: List[Dict[str, int]] = [dict() for _ in range(text_length + 1)]
    # Sentinel empty last token for the start position
    dp[0] = {"": 0}

    for i in range(1, text_length + 1):
        best_end_here: Dict[str, int] = {}
        for length in token_length_set:
            j = i - length
            if j < 0:
                continue
            candidate_token = text[j:i]
            if candidate_token not in vocab_scores:
                continue
            base_score = vocab_scores[candidate_token]
            prev_map = dp[j]
            if not prev_map:
                continue
            for prev_token, prev_score in prev_map.items():
                bonus = 0 if prev_token == "" else transitions.get((prev_token, candidate_token), 0)
                candidate_score = prev_score + base_score + bonus
                current_best = best_end_here.get(candidate_token)
                if current_best is None or candidate_score > current_best:
                    best_end_here[candidate_token] = candidate_score
        dp[i] = best_end_here

    if not dp[text_length]:
        return 0
    return max(dp[text_length].values())


def main() -> None:
    text, vocab_scores, transitions = parse_input_from_stdin()
    best_score = compute_best_segmentation_score(text, vocab_scores, transitions)
    print(best_score)


if __name__ == "__main__":
    main()

