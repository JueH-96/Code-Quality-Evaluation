import sys
import threading

def main():
    import sys
    input = sys.stdin.readline

    N, M = map(int, input().split())
    intervals = []
    for i in range(M):
        l, r = map(int, input().split())
        intervals.append((l, r, i))

    # Sort by L, then by R descending
    intervals_sorted = sorted(intervals, key=lambda x: (x[0], -x[1]))

    # 1. Try to find a disjoint pair -> B of size 2, cost = 2
    max_r = -1
    idx_max_r = -1
    disj_pair = None
    for l, r, idx in intervals_sorted:
        if max_r < l:
            # We found an earlier interval with R = max_r < this L => disjoint
            disj_pair = (idx_max_r, idx)
            break
        if r > max_r:
            max_r = r
            idx_max_r = idx

    # 2. Compute greedy cover_cost for [1..N]
    cover_ops = []
    cover_cost = 0
    cur = 1
    i = 0
    best_r = -1
    best_idx = -1
    while cur <= N:
        # advance while intervals_sorted[i].L <= cur
        while i < M and intervals_sorted[i][0] <= cur:
            l, r, idx = intervals_sorted[i]
            if r > best_r:
                best_r = r
                best_idx = idx
            i += 1
        # pick the best
        if best_r < cur:
            # cannot cover
            cover_cost = float('inf')
            cover_ops = []
            break
        # use that interval
        cover_cost += 1
        cover_ops.append(best_idx)
        cur = best_r + 1
        # reset best
        best_r = -1
        best_idx = -1

    # Decide which strategy to use
    # strategy A: cover_cost via op1, cost = cover_cost
    # strategy B: disjoint pair via op2, cost = 2 (only if disj_pair exists)
    use_cover = False
    use_disjoint = False

    if cover_cost == float('inf') and disj_pair is None:
        print(-1)
        return

    # pick minimal
    best_cost = float('inf')
    if cover_cost < best_cost:
        best_cost = cover_cost
        use_cover = True
    if disj_pair is not None and 2 < best_cost:
        best_cost = 2
        use_disjoint = True
        use_cover = False

    # Build operations
    ops = [0] * M
    if use_cover:
        # mark cover_ops with op = 1
        for idx in cover_ops:
            ops[idx] = 1
    elif use_disjoint:
        i1, i2 = disj_pair
        ops[i1] = 2
        ops[i2] = 2
    else:
        # Edge: if cover_cost == 2 == disjoint exists, tie picks cover by code above
        # Already handled.

        pass

    # Print
    print(best_cost)
    print(" ".join(str(o) for o in ops))


if __name__ == "__main__":
    main()