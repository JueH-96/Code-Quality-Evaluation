def solve():
    N, K = map(int, input().split())
    A = list(map(int, input().split()))
    MOD = 998244353
    
    # dp[i][0] = number of ways to partition first i elements with no subsequence summing to K
    # dp[i][1] = number of ways to partition first i elements with at least one subsequence summing to K
    dp = [[0, 0] for _ in range(N + 1)]
    dp[0][0] = 1  # Base case: empty sequence has no subsequence with sum K
    
    # Compute prefix sums for efficient range sum queries
    prefix_sum = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_sum[i] = prefix_sum[i - 1] + A[i - 1]
    
    for i in range(1, N + 1):
        for j in range(1, i + 1):
            # Calculate sum of subsequence from j to i
            subseq_sum = prefix_sum[i] - prefix_sum[j - 1]
            
            if subseq_sum == K:
                # If we found a subsequence with sum K, it contributes to the "at least one" case
                dp[i][1] = (dp[i][1] + dp[j-1][0] + dp[j-1][1]) % MOD
            else:
                # If this subsequence doesn't sum to K:
                # - If previous has no subsequence with sum K, we maintain that property
                dp[i][0] = (dp[i][0] + dp[j-1][0]) % MOD
                # - If previous has at least one subsequence with sum K, we maintain that property
                dp[i][1] = (dp[i][1] + dp[j-1][1]) % MOD
    
    return dp[N][0]

if __name__ == "__main__":
    print(solve())