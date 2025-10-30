# YOUR CODE HERE
import sys

def min_time_to_play_stage_n(N, stages):
    dp = [float('inf')] * (N + 1)
    dp[1] = 0

    for i in range(1, N):
        a, b, x = stages[i-1]
        dp[i+1] = min(dp[i+1], dp[i] + a)
        dp[x] = min(dp[x], dp[i] + b)

    return dp[N]

N = int(input())
stages = [tuple(map(int, input().split())) for _ in range(N-1)]

print(min_time_to_play_stage_n(N, stages))