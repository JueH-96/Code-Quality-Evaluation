def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())

    # Precompute repunits: 1, 11, 111, ..., up to length L
    L = 20
    repunits = []
    cur = 0
    for _ in range(L):
        cur = cur * 10 + 1
        repunits.append(cur)

    # Compute all sums of exactly three repunits (with repetition allowed)
    sums = set()
    n = len(repunits)
    for i in range(n):
        ri = repunits[i]
        for j in range(i, n):
            rj = repunits[j]
            for k in range(j, n):
                rk = repunits[k]
                sums.add(ri + rj + rk)

    # Sort the unique sums and pick the N-th smallest (1-based)
    sorted_sums = sorted(sums)
    print(sorted_sums[N-1])


if __name__ == "__main__":
    main()