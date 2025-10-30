def solve():
    N, K = map(int, input().split())
    A = list(map(int, input().split()))
    
    MOD = 998244353
    
    # dp[i] = number of valid ways to divide A[0:i]
    dp = [0] * (N + 1)
    dp[0] = 1  # empty sequence has one way
    
    # For each position i
    for i in range(1, N + 1):
        # Try all possible starting positions j for the last subsequence
        current_sum = 0
        for j in range(i - 1, -1, -1):
            # Add A[j] to get sum of subsequence A[j:i]
            current_sum += A[j]
            
            # If this subsequence doesn't sum to K, we can use it
            if current_sum != K:
                dp[i] = (dp[i] + dp[j]) % MOD
    
    print(dp[N])

solve()