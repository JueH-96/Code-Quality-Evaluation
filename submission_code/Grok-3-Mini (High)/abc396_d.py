import sys

# Read all input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1

# Create adjacency list
adj = [[] for _ in range(N + 1)]
for _ in range(M):
    u = int(data[index])
    v = int(data[index + 1])
    w = int(data[index + 2])
    index += 3
    adj[u].append((v, w))
    adj[v].append((u, w))

# Initialize minimum XOR to a large value
min_xor = (1 << 60)  # 2^60, larger than any possible XOR value

def dfs(node, mask, xor_sum):
    global min_xor
    if node == N:
        if xor_sum < min_xor:
            min_xor = xor_sum
        return
    for neigh, weight in adj[node]:
        if (mask & (1 << (neigh - 1))) == 0:  # Neighbor not visited
            new_mask = mask | (1 << (neigh - 1))
            dfs(neigh, new_mask, xor_sum ^ weight)

# Start DFS from node 1 with mask indicating node 1 is visited and XOR sum 0
dfs(1, 1 << 0, 0)

# Output the minimum XOR
print(min_xor)