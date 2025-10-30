import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
W = int(data[1])
items = []

index = 2
for i in range(N):
    w = int(data[index])
    v = int(data[index + 1])
    items.append((w, v))
    index += 2

# Dynamic programming table
dp = [0] * (W + 1)

for w, v in items:
    for j in range(W, w - 1, -1):
        for k in range(1, min(100, j // w) + 1):
            happiness = k * v - k * k
            if happiness > 0:
                dp[j] = max(dp[j], dp[j - k * w] + happiness)

print(max(dp))