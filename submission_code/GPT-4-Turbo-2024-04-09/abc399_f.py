def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    MOD = 998244353
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))
    
    # Precompute powers of each A[i] up to K
    power = [[1] * (K + 1) for _ in range(N)]
    for i in range(N):
        for k in range(1, K + 1):
            power[i][k] = power[i][k - 1] * A[i] % MOD
    
    # Sum of powers of sums
    result = 0
    
    # We use a dynamic programming approach to calculate the sum of powers of subarray sums
    # dp[k][r] will be the sum of (sum of A[l] to A[r])^k for all l <= r
    dp = [[0] * (N + 1) for _ in range(K + 1)]
    
    # Initialize for k = 0, (any sum to the power of 0 is 1)
    for r in range(1, N + 1):
        dp[0][r] = 1
    
    # Fill dp for k = 1 to K
    for k in range(1, K + 1):
        # We use a prefix sum to efficiently calculate range sums
        prefix_sum = [0] * (N + 1)
        for r in range(1, N + 1):
            prefix_sum[r] = (prefix_sum[r - 1] + A[r - 1]) % MOD
        
        # We also need cumulative sums for dp[k-1][j] for efficient computation
        dp_cumulative = [0] * (N + 1)
        for r in range(1, N + 1):
            dp_cumulative[r] = (dp_cumulative[r - 1] + dp[k - 1][r]) % MOD
        
        # Calculate dp[k][r]
        for r in range(1, N + 1):
            dp[k][r] = (dp_cumulative[r] * prefix_sum[r]) % MOD
            if r > 0:
                dp[k][r] = (dp[k][r] - dp_cumulative[r - 1] * prefix_sum[r - 1]) % MOD
                dp[k][r] = (dp[k][r] + MOD) % MOD
    
    # Sum up all dp[K][r] for r from 1 to N
    for r in range(1, N + 1):
        result = (result + dp[K][r]) % MOD
    
    print(result)

if __name__ == "__main__":
    main()