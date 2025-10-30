# YOUR CODE HERE
MOD = 998244353

def solve(N, A):
    dp = [0] * (N + 1)
    dp[0] = 1
    
    for i in range(1, N + 1):
        prefix_sum = [0] * (i + 1)
        for j in range(i - 1, A[i-1] - 1, -1):
            prefix_sum[j + 1] = (prefix_sum[j + 1] + dp[j]) % MOD
        
        for j in range(1, i + 1):
            prefix_sum[j] = (prefix_sum[j] + prefix_sum[j - 1]) % MOD
        
        dp[i] = prefix_sum[i] - prefix_sum[A[i-1]]
        if dp[i] < 0:
            dp[i] += MOD
    
    return dp[N]

# Read input
N = int(input())
A = list(map(int, input().split()))

# Solve and print result
result = solve(N, A)
print(result)