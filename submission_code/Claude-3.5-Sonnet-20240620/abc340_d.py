# YOUR CODE HERE
import sys

def min_time_to_reach_stage_n(n, stages):
    dp = [float('inf')] * (n + 1)
    dp[1] = 0

    for i in range(1, n):
        a, b, x = stages[i-1]
        dp[i+1] = min(dp[i+1], dp[i] + a)
        dp[x] = min(dp[x], dp[i] + b)

    return dp[n]

n = int(sys.stdin.readline().strip())
stages = []
for _ in range(n-1):
    a, b, x = map(int, sys.stdin.readline().split())
    stages.append((a, b, x))

result = min_time_to_reach_stage_n(n, stages)
print(result)