def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    A = list(map(int, data[2:]))

    MOD = 998244353
    
    # Compute prefix sums S, where S[i] = A[0] + A[1] + ... + A[i-1]
    S = [0] * (N + 1)
    for i in range(N):
        S[i+1] = S[i] + A[i]
    
    # dp[i] = number of valid ways to partition the prefix A[0..i-1]
    # prefix_dp[i] = dp[0] + dp[1] + ... + dp[i]
    dp = [0] * (N + 1)
    dp[0] = 1
    prefix_dp = [0] * (N + 1)
    prefix_dp[0] = 1
    
    # counts[x] = sum of dp[j] for all j where S[j] = x
    counts = {0: 1}
    
    for i in range(1, N + 1):
        # Sum of dp[j] for j = 0..i-1 minus those j where S[i] - S[j] == K
        # i.e. minus dp[j] for all j such that S[j] = S[i] - K
        val = prefix_dp[i-1] - counts.get(S[i] - K, 0)
        val %= MOD
        
        dp[i] = val
        prefix_dp[i] = (prefix_dp[i-1] + dp[i]) % MOD
        
        # Update counts for this prefix sum
        counts[S[i]] = (counts.get(S[i], 0) + dp[i]) % MOD
    
    # dp[N] is the number of valid partitions of the entire array
    print(dp[N] % MOD)

# Do not forget to call main()
if __name__ == "__main__":
    main()