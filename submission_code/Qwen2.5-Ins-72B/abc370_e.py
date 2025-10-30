# YOUR CODE HERE
import sys
from collections import defaultdict

MOD = 998244353

def solve(n, k, a):
    dp = [0] * (n + 1)
    dp[0] = 1
    prefix_sum = 0
    count = defaultdict(int)
    count[0] = 1
    
    for i in range(n):
        prefix_sum += a[i]
        dp[i + 1] = (2 * dp[i] - count[prefix_sum - k]) % MOD
        count[prefix_sum] += dp[i]
    
    return dp[n]

input = sys.stdin.read
data = input().split()
n, k = int(data[0]), int(data[1])
a = list(map(int, data[2:]))
print(solve(n, k, a))