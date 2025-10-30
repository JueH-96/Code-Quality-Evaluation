import sys
sys.setrecursionlimit(1000000)

# Read all input from stdin
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1

# Build adjacency list
adj = [[] for _ in range(N + 1)]
for _ in range(N - 1):
    u = int(data[index])
    v = int(data[index + 1])
    index += 2
    adj[u].append(v)
    adj[v].append(u)

# DFS function to compute subtree size
def dfs(node, parent):
    size = 1
    for nei in adj[node]:
        if nei != parent:
            size += dfs(nei, node)
    return size

# Get the children of root (vertex 1)
children = adj[1]
k = len(children)

# Compute the size of each subtree rooted at the children
sizes = []
for child in children:
    sz_child = dfs(child, 1)  # Parent is 1 for all children of root
    sizes.append(sz_child)

# Sort the sizes
sizes.sort()

# Sum of the smallest k-1 subtree sizes
min_sum = sum(sizes[:k - 1])

# Minimum operations to delete vertex 1
ans = 1 + min_sum

# Output the answer
print(ans)