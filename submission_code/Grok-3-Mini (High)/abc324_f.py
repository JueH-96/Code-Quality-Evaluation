import sys
import math

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1

# Store incoming edges for each vertex
incoming = [[] for _ in range(N + 1)]
for _ in range(M):
    u = int(data[index])
    v = int(data[index + 1])
    beauty = int(data[index + 2])
    cost = int(data[index + 3])
    index += 4
    incoming[v].append((u, beauty, cost))

# Function to compute the maximum sum of (beauty - R * cost) from vertex 1 to vertex N
def max_sum(R):
    dp = [float('-inf')] * (N + 1)
    dp[1] = 0.0
    for v in range(2, N + 1):
        for edge in incoming[v]:
            u, b, c = edge
            if dp[u] != float('-inf'):
                dp[v] = max(dp[v], dp[u] + (b - R * c))
    return dp[N]

# Binary search to find the maximum ratio
low = 0.0
high = 10000.0  # Maximum possible ratio since beauty and cost <= 10000
for _ in range(60):  # 60 iterations for precision
    mid = (low + high) / 2
    if max_sum(mid) >= 0.0:
        low = mid
    else:
        high = mid

# Output the result with high precision
print("{:.15f}".format(low))