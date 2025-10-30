# YOUR CODE HERE
from collections import defaultdict

def solve():
    mod = 998244353
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    prefix_sum = [0]
    for x in a:
        prefix_sum.append(prefix_sum[-1] + x)
    
    dp = [0] * (n + 1)
    dp[0] = 1
    prefix_sum_count = defaultdict(int)
    prefix_sum_count[0] = 1
    
    for i in range(1, n + 1):
        dp[i] = dp[i - 1] * 2 % mod
        if prefix_sum[i] - k in prefix_sum_count:
            dp[i] -= prefix_sum_count[prefix_sum[i] - k]
        dp[i] %= mod
        prefix_sum_count[prefix_sum[i]] += dp[i]
    
    print(dp[n])

solve()