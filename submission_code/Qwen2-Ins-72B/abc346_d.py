from itertools import pairwise
from math import inf

N = int(input())
S = input()
C = list(map(int, input().split()))

costs = [0] * (N - 1)
for i, (s1, s2) in enumerate(pairwise(S)):
    if s1 == s2:
        costs[i] = inf
    else:
        costs[i] = min(C[i], C[i + 1])

dp = [0] * (N + 1)
for i, cost in enumerate(costs):
    dp[i + 2] = min(dp[i] + cost, dp[i + 1])

dp[-1] = min(dp[-1], dp[-2] + C[-1])
print(dp[-1])