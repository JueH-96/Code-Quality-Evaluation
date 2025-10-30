n = int(input())

# Build adjacency list
adj = [[] for _ in range(n + 1)]
for _ in range(n - 1):
    a, b = map(int, input().split())
    adj[a].append(b)
    adj[b].append(a)

c = [0] + list(map(int, input().split()))  # 1-indexed

# DFS to calculate subtree sums and f[1]
subtree_sum = [0] * (n + 1)

def dfs1(u, parent, depth):
    subtree_sum[u] = c[u]
    result = c[u] * depth
    for v in adj[u]:
        if v != parent:
            result += dfs1(v, u, depth + 1)
            subtree_sum[u] += subtree_sum[v]
    return result

f1 = dfs1(1, -1, 0)

# Calculate total sum
total_sum = sum(c[1:])

# DFS to calculate f(v) for all vertices
f = [0] * (n + 1)
f[1] = f1

def dfs2(u, parent):
    for v in adj[u]:
        if v != parent:
            f[v] = f[u] + total_sum - 2 * subtree_sum[v]
            dfs2(v, u)

dfs2(1, -1)

# Find and print minimum
print(min(f[1:]))