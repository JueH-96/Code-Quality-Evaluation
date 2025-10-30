def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    # Read weights into a N x N matrix (0-indexed)
    w = [[0] * N for _ in range(N)]
    for i in range(N - 1):
        for j in range(i + 1, N):
            val = int(next(it))
            w[i][j] = val
            w[j][i] = val

    FULL = (1 << N) - 1
    dp = [0] * (1 << N)
    vis = [False] * (1 << N)

    def solve(mask):
        if vis[mask]:
            return dp[mask]
        vis[mask] = True
        if mask == 0:
            dp[mask] = 0
            return 0
        # pick the lowest-numbered vertex in mask
        lowbit = mask & -mask
        i = lowbit.bit_length() - 1
        rest = mask ^ lowbit
        best = solve(rest)  # skip i (leave it unmatched)
        mm = rest
        # try pairing i with any other j in rest
        while mm:
            lb = mm & -mm
            j = lb.bit_length() - 1
            cand = solve(rest ^ lb) + w[i][j]
            if cand > best:
                best = cand
            mm ^= lb
        dp[mask] = best
        return best

    ans = solve(FULL)
    print(ans)

if __name__ == "__main__":
    main()