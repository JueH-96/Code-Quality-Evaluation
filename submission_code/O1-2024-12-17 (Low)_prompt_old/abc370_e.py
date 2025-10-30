def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    K = int(input_data[1])
    A = list(map(int, input_data[2:]))

    MOD = 998244353

    # Compute prefix sums
    prefix_sum = [0] * (N+1)
    for i in range(N):
        prefix_sum[i+1] = prefix_sum[i] + A[i]

    # dp[i] = number of ways to partition A[:i] such that no block sums to K
    # We'll also keep track of prefixDP[i] = dp[0] + dp[1] + ... + dp[i]
    dp = [0]*(N+1)
    prefixDP = [0]*(N+1)

    # freq[ s ] = sum of dp[j] for all j where prefix_sum[j] = s
    freq = {}

    # Base case
    dp[0] = 1
    prefixDP[0] = 1
    freq[0] = 1  # prefix_sum[0] = 0, dp[0] = 1

    for i in range(N):
        s_new = prefix_sum[i+1]
        want = s_new - K
        
        # dp[i+1] = prefixDP[i] - sum_of_dp_j_where_prefix_sum_j == want
        ways_to_exclude = freq.get(want, 0)
        val = prefixDP[i] - ways_to_exclude
        val %= MOD  # handle negativity

        dp[i+1] = val
        prefixDP[i+1] = (prefixDP[i] + dp[i+1]) % MOD

        # Update freq
        freq[s_new] = (freq.get(s_new, 0) + dp[i+1]) % MOD

    print(dp[N] % MOD)