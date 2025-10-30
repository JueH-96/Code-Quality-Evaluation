def main() -> None:
    import sys
    from itertools import combinations

    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))

    masks = []
    for _ in range(N):
        s = next(it)
        mask = 0
        for j, ch in enumerate(s):
            if ch == 'o':
                mask |= 1 << j
        masks.append(mask)

    full_mask = (1 << M) - 1
    best = N + 1  # larger than any possible answer

    # Enumerate all subsets of stands
    for subset in range(1, 1 << N):
        if subset.bit_count() >= best:
            continue  # cannot improve current best
        cur = 0
        for i in range(N):
            if subset >> i & 1:
                cur |= masks[i]
        if cur == full_mask:
            best = subset.bit_count()
            if best == 1:  # can't do better than 1
                break

    print(best)


if __name__ == "__main__":
    main()