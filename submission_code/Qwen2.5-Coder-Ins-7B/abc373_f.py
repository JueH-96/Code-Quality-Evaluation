import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
W = int(data[1])

items = []
for i in range(N):
    w = int(data[2 + 2 * i])
    v = int(data[3 + 2 * i])
    items.append((w, v))

dp = [0] * (W + 1)

for w, v in items:
    for j in range(W, w - 1, -1):
        dp[j] = max(dp[j], dp[j - w] + v - (j - w) ** 2)

print(dp[W])