import math
import sys

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
x = [0] * (N + 1)
y = [0] * (N + 1)
for i in range(1, N + 1):
    x[i] = int(data[index])
    y[i] = int(data[index + 1])
    index += 2

# Initialize dp: list of dictionaries
dp = [{} for _ in range(N + 1)]

# Set for k=1
max_i_k1 = min(N, 29)  # 1 + 28 = 29
for i in range(1, max_i_k1 + 1):
    if i == 1:
        dp[i][1] = 0.0
    else:
        dp[i][1] = float('inf')

# For k from 2 to N
for k in range(2, N + 1):
    min_i = k
    max_i = min(N, k + 28)
    for i in range(min_i, max_i + 1):
        min_val = float('inf')
        j_low = max(1, k - 1)
        j_high = min(N, (k - 1) + 28)
        for j in range(j_low, j_high + 1):
            if j < i:
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                dist_ji = math.sqrt(dx * dx + dy * dy)
                val = dp[j].get(k - 1, float('inf')) + dist_ji
                if val < min_val:
                    min_val = val
        dp[i][k] = min_val

# Compute the minimum s
min_s = float('inf')
for k in dp[N]:  # Iterate over the keys in dp[N]
    dist_cost = dp[N][k]
    C = N - k
    if C == 0:
        pen = 0.0
    else:
        pen = math.pow(2.0, C - 1)
    s = dist_cost + pen
    if s < min_s:
        min_s = s

# Output the result with high precision
print(f"{min_s:.15f}")