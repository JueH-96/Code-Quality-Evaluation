import sys

# Read all input
data = list(map(int, sys.stdin.read().split()))
index = 0
N = data[index]
index += 1
M = data[index]
index += 1

# Read edges and build adjacency list
adj = [[] for _ in range(N + 1)]
for _ in range(M):
    u = data[index]
    index += 1
    v = data[index]
    index += 1
    adj[u].append(v)
    adj[v].append(u)

# Read weights W
W = [0] * (N + 1)
for i in range(1, N + 1):
    W[i] = data[index]
    index += 1

# Read initial pieces A
A = [0] * (N + 1)
for i in range(1, N + 1):
    A[i] = data[index]
    index += 1

# Sort vertices by weight in ascending order
verts = list(range(1, N + 1))
verts.sort(key=lambda x: W[x])

# Array to store val for each vertex
val_arr = [0] * (N + 1)

# Knapsack function to compute max value with weight <= cap
def compute_knapsack(items, cap):
    if cap < 0:
        return 0  # Cap should not be negative, but handle just in case
    dp = [0] * (cap + 1)
    for weight, value in items:
        if weight > cap:
            continue  # Skip item that exceeds capacity
        for w in range(cap, weight - 1, -1):  # Update dp from cap downto weight
            dp[w] = max(dp[w], dp[w - weight] + value)
    return max(dp)  # Max value over all weights <= cap

# Compute val for each vertex in ascending weight order
for x in verts:
    # Find out-neighbors: adjacent and W[y] < W[x]
    out_neighbors = [y for y in adj[x] if W[y] < W[x]]
    # Create items for knapsack: (weight W[y], value val[y])
    items = [(W[y], val_arr[y]) for y in out_neighbors]
    # Capacity for knapsack
    cap = W[x] - 1
    # Compute max sum of val[y] with sum W[y] <= cap
    max_sum_val = compute_knapsack(items, cap)
    # Set val[x]
    val_arr[x] = 1 + max_sum_val

# Compute the total sum A[i] * val[i]
total = 0
for i in range(1, N + 1):
    total += A[i] * val_arr[i]

# Output the result
print(total)