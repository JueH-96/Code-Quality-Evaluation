# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:]))
    
    MOD = 998244353
    
    # dp[i][v] will store the number of valid sequences for the first i elements with x_i = v
    dp = [[0] * (M + 1) for _ in range(N + 1)]
    
    # Base case: for the first element, any value from 1 to M is valid
    for v in range(1, M + 1):
        dp[1][v] = 1
    
    # Fill dp table
    for i in range(2, N + 1):
        # Compute prefix sums for dp[i-1] to efficiently calculate ranges
        prefix_sum = [0] * (M + 1)
        for v in range(1, M + 1):
            prefix_sum[v] = (prefix_sum[v - 1] + dp[i - 1][v]) % MOD
        
        for v in range(1, M + 1):
            # We need to ensure x_i <= x_{A_i}, which means v <= x_{A_i}
            # x_{A_i} is in the range [v, M]
            if A[i - 1] < i:
                dp[i][v] = (prefix_sum[M] - prefix_sum[v - 1]) % MOD
            else:
                dp[i][v] = dp[i - 1][v]
    
    # Calculate the total number of valid sequences
    result = sum(dp[N][v] for v in range(1, M + 1)) % MOD
    
    print(result)