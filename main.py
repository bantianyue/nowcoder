import sys


def main() -> None:
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(map(int, data))

    try:
        n = next(it)
        m = next(it)
        k = next(it)
    except StopIteration:
        return

    # Node storage: 1-based indexing
    left = [0] * (n + 1)
    right = [0] * (n + 1)
    feat = [0] * (n + 1)
    thr = [0] * (n + 1)
    cls = [0] * (n + 1)

    for i in range(1, n + 1):
        try:
            l = next(it)
            r = next(it)
            f = next(it)
            t = next(it)
            c = next(it)
        except StopIteration:
            return
        left[i] = l
        right[i] = r
        feat[i] = f
        thr[i] = t
        cls[i] = c

    # Count positives/negatives reaching each node
    pos = [0] * (n + 1)
    neg = [0] * (n + 1)

    for _ in range(m):
        x = [0] * k
        for j in range(k):
            try:
                x[j] = next(it)
            except StopIteration:
                return
        try:
            y = next(it)
        except StopIteration:
            return

        u = 1
        while True:
            if y == 1:
                pos[u] += 1
            else:
                neg[u] += 1
            if left[u] == 0 and right[u] == 0:
                break
            f = feat[u]
            t = thr[u]
            if x[f - 1] <= t:
                u = left[u]
            else:
                u = right[u]

    # Build postorder list (children before parent)
    post = []
    stack = [(1, 0)]
    while stack:
        u, vis = stack.pop()
        if u == 0:
            continue
        if vis:
            post.append(u)
        else:
            stack.append((u, 1))
            if right[u]:
                stack.append((right[u], 0))
            if left[u]:
                stack.append((left[u], 0))

    best = [0.0] * (n + 1)

    def feasible(F: float) -> bool:
        # Tree DP: for each node choose either prune here (take v_u) or keep children
        for u in post:
            if cls[u] == 1:
                vu = (2.0 - 2.0 * F) * float(pos[u]) - F * float(neg[u])
            else:
                vu = -F * float(pos[u])
            if left[u] == 0 and right[u] == 0:
                best[u] = vu
            else:
                best[u] = vu if vu >= best[left[u]] + best[right[u]] else best[left[u]] + best[right[u]]
        return best[1] >= 0.0

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) * 0.5
        if feasible(mid):
            lo = mid
        else:
            hi = mid

    sys.stdout.write(f"{lo:.6f}\n")


if __name__ == "__main__":
    main()

