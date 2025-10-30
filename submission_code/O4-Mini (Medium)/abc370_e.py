def main():
    import sys
    input = sys.stdin.readline
    MOD = 998244353

    N, K = map(int, input().split())
    A = list(map(int, input().split()))

    # dp[r]: number of ways to place cuts so that the last cut is at position r.
    # We only need dp[r] for the current r and prefix sums of dp.
    # Let S[r] = prefix sum up to r. Then:
    # dp[r] = (sum_{l=0..r-1} dp[l]) - (sum of dp[l] over l<r with S[l] = S[r] - K)
    # Maintain total = sum_{l=0..r-1} dp[l], and a map M from prefix sum value to sum of dp[l] with that prefix.
    # Initialize dp[0] = 1, S[0]=0, total=1, M[0]=1

    total = 1          # total = sum dp[0..r-1]; at start r=0, total = dp[0] = 1
    prefix = 0         # S[0]
    M = {0: 1}         # M[s] = sum of dp[l] for which S[l] == s
    dp_r = 0           # placeholder for dp[r]

    for x in A:
        prefix += x
        # total is sum dp[0..r-1]
        bad = M.get(prefix - K, 0)
        dp_r = (total - bad) % MOD
        total = (total + dp_r) % MOD
        M[prefix] = (M.get(prefix, 0) + dp_r) % MOD

    # dp_r at the last iteration is dp[N]
    print(dp_r)

if __name__ == "__main__":
    main()