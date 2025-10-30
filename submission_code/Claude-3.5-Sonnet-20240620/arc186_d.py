# YOUR CODE HERE
MOD = 998244353

def count_polish_sequences(N, A):
    dp = [0] * (N + 1)
    dp[0] = 1
    
    prefix_sum = [0] * (N + 1)
    prefix_sum[0] = 1
    
    for i in range(1, N + 1):
        if i - 1 >= A[i - 1]:
            dp[i] = prefix_sum[i - 1] - prefix_sum[i - 1 - A[i - 1]]
        else:
            dp[i] = prefix_sum[i - 1]
        
        dp[i] %= MOD
        prefix_sum[i] = (prefix_sum[i - 1] + dp[i]) % MOD

    return dp[N]

# Read input
N = int(input())
A = list(map(int, input().split()))

# Solve and print output
result = count_polish_sequences(N, A)
print(result)