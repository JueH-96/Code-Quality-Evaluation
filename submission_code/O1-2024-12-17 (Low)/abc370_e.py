def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, K = map(int, input_data[:2])
    A = list(map(int, input_data[2:]))

    MOD = 998244353

    # Prefix sums: S[i] = A[1] + A[2] + ... + A[i], with S[0] = 0.
    # dp[i] = number of valid ways to partition A[:i] (1-based indexing).
    # dp[0] = 1 (base case: one way to partition empty prefix).
    # prefix_dp[i] = dp[0] + dp[1] + ... + dp[i] (for fast cumulative sums).
    # We also keep a hashmap sumDP[x] = sum of all dp[j] for j where S[j] = x.
    # Recurrence:
    #   dp[i] = prefix_dp[i-1] - sumDP[S[i] - K],
    #   then prefix_dp[i] = prefix_dp[i-1] + dp[i],
    #   and sumDP[S[i]] += dp[i].
    #
    # The answer will be dp[N] mod 998244353.

    prefix_sums = [0] * (N + 1)
    for i in range(N):
        prefix_sums[i + 1] = prefix_sums[i] + A[i]

    dp = [0] * (N + 1)
    prefix_dp = [0] * (N + 1)
    sumDP = {}

    dp[0] = 1
    prefix_dp[0] = 1
    sumDP[prefix_sums[0]] = sumDP.get(prefix_sums[0], 0) + 1  # sumDP[0] = 1 initially

    for i in range(1, N + 1):
        # Compute dp[i]
        total_ways = prefix_dp[i - 1]
        need = prefix_sums[i] - K
        forbidden = sumDP.get(need, 0)
        ways = (total_ways - forbidden) % MOD

        dp[i] = ways
        prefix_dp[i] = (prefix_dp[i - 1] + dp[i]) % MOD

        # Update sumDP for prefix_sums[i]
        cur_sum = prefix_sums[i]
        sumDP[cur_sum] = (sumDP.get(cur_sum, 0) + dp[i]) % MOD

    print(dp[N] % MOD)

# Do not forget to call main()
if __name__ == "__main__":
    main()