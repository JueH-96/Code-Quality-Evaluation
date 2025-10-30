import sys
import heapq
from collections import defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

index = 2
wheels = []

for i in range(N):
    C = int(data[index])
    P = int(data[index + 1])
    scores = list(map(int, data[index + 2: index + 2 + P]))
    index += 2 + P
    wheels.append((C, P, scores))

# dp[m] will hold the minimum expected cost to reach at least m points
dp = [float('inf')] * (M + 1)
dp[0] = 0.0  # No cost to reach 0 points

for m in range(M):
    if dp[m] == float('inf'):
        continue
    for C, P, scores in wheels:
        expected_cost = dp[m] + C
        for score in scores:
            new_points = min(m + score, M)
            if dp[new_points] > expected_cost:
                dp[new_points] = expected_cost

print(dp[M])