# YOUR CODE HERE
import sys
input = sys.stdin.read
from math import isqrt

def max_happiness(N, W, weights, values):
    dp = [0] * (W + 1)
    for w, v in zip(weights, values):
        for j in range(W, w - 1, -1):
            k = isqrt(j // w)
            dp[j] = max(dp[j], dp[j - w * k] + k * v - k * k)
    return dp[W]

N, W = map(int, input().split())
weights = []
values = []
for _ in range(N):
    w, v = map(int, input().split())
    weights.append(w)
    values.append(v)
print(max_happiness(N, W, weights, values))