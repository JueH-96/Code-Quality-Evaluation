from collections import defaultdict

n = int(input())

# Build adjacency list
adj = defaultdict(list)
for _ in range(n - 1):
    a, b = map(int, input().split())
    adj[a].append(b)
    adj[b].append(a)

c = list(map(int, input().split()))
total_sum = sum(c)

# First DFS: compute subtree sums
subtree_sum = [0] * (n + 1)

def dfs1(node, parent):
    subtree_sum[node] = c[node - 1]
    for neighbor in adj[node]:
        if neighbor != parent:
            dfs1(neighbor, node)
            subtree_sum[node] += subtree_sum[neighbor]

dfs1(1, -1)

# Compute f(1)
f = [0] * (n + 1)

def compute_initial_f(node, parent, depth):
    f[1] += c[node - 1] * depth
    for neighbor in adj[node]:
        if neighbor != parent:
            compute_initial_f(neighbor, node, depth + 1)

compute_initial_f(1, -1, 0)

# Second DFS: reroot to compute f(v) for all v
def dfs2(node, parent):
    for neighbor in adj[node]:
        if neighbor != parent:
            # Move from node to neighbor
            f[neighbor] = f[node] + total_sum - 2 * subtree_sum[neighbor]
            dfs2(neighbor, node)

dfs2(1, -1)

print(min(f[1:n+1]))