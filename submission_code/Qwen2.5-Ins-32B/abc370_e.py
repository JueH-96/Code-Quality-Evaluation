import sys
from collections import defaultdict

def solve():
    MOD = 998244353
    N, K = map(int, input().split())
    A = list(map(int, input().split()))
    
    dp = [0] * (N + 1)
    dp[0] = 1
    prefix_sum = 0
    sum_counts = defaultdict(int)
    sum_counts[0] = 1
    
    for i in range(1, N + 1):
        prefix_sum += A[i - 1]
        dp[i] = (2 * dp[i - 1]) % MOD
        
        if prefix_sum - K in sum_counts:
            dp[i] -= sum_counts[prefix_sum - K]
            dp[i] %= MOD
        
        sum_counts[prefix_sum] += dp[i - 1]
        sum_counts[prefix_sum] %= MOD
    
    print(dp[N])

solve()