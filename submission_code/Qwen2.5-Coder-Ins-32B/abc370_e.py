# YOUR CODE HERE
MOD = 998244353

def count_valid_partitions(N, K, A):
    dp = [0] * (N + 1)
    dp[0] = 1
    prefix_sum = 0
    prefix_sum_count = {0: 1}
    
    for i in range(1, N + 1):
        prefix_sum += A[i - 1]
        dp[i] = (2 * dp[i - 1]) % MOD
        
        if (prefix_sum - K) in prefix_sum_count:
            dp[i] = (dp[i] - prefix_sum_count[prefix_sum - K] + MOD) % MOD
        
        if prefix_sum in prefix_sum_count:
            prefix_sum_count[prefix_sum] += dp[i - 1]
        else:
            prefix_sum_count[prefix_sum] = dp[i - 1]
    
    return dp[N]

import sys
input = sys.stdin.read().split()
N = int(input[0])
K = int(input[1])
A = list(map(int, input[2:]))

print(count_valid_partitions(N, K, A))