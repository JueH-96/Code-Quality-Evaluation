# YOUR CODE HERE
import sys
from math import inf

input = sys.stdin.read
data = input().split()
N = int(data[0])
S = int(data[1])
M = int(data[2])
L = int(data[3])

dp = [0] + [inf] * N

for i in range(6, N + 1):
    dp[i] = min(dp[i], dp[i - 6] + S)
for i in range(8, N + 1):
    dp[i] = min(dp[i], dp[i - 8] + M)
for i in range(12, N + 1):
    dp[i] = min(dp[i], dp[i - 12] + L)

print(dp[N])