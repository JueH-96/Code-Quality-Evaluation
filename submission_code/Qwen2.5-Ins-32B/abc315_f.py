import sys
import math

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def min_distance_with_penalty(N, checkpoints):
    dp = [float('inf')] * (N + 1)
    dp[1] = 0
    for i in range(2, N + 1):
        for j in range(1, i):
            dist = distance(checkpoints[j - 1], checkpoints[i - 1])
            penalty = 2 ** (i - j - 2) if i - j > 1 else 0
            dp[i] = min(dp[i], dp[j] + dist + penalty)
    return dp[N]

N = int(input())
checkpoints = [tuple(map(int, input().split())) for _ in range(N)]
print(min_distance_with_penalty(N, checkpoints))