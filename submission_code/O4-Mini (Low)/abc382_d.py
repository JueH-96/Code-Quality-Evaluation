def main():
    import sys
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])

    results = []
    seq = []

    # Recursive backtracking
    def dfs(pos, prev):
        # pos is the 1-based position we are filling
        if pos > N:
            # Completed a full sequence
            results.append(seq.copy())
            return
        # Determine lower bound for this position
        if pos == 1:
            lo = 1
        else:
            lo = prev + 10
        # Determine upper bound so that we can still fit the remaining positions
        # If we choose x here, the maximum for A_N is x + 10*(N-pos),
        # which must be <= M
        hi = M - 10 * (N - pos)
        if lo > hi:
            return
        for x in range(lo, hi + 1):
            seq.append(x)
            dfs(pos + 1, x)
            seq.pop()

    dfs(1, 0)

    # Output
    out = []
    out.append(str(len(results)))
    for r in results:
        out.append(" ".join(str(x) for x in r))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()