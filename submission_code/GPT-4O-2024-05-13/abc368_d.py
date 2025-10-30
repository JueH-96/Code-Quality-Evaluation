# YOUR CODE HERE
import sys
import collections

input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])

edges = []
index = 2
for _ in range(N-1):
    A = int(data[index])
    B = int(data[index+1])
    edges.append((A, B))
    index += 2

V = list(map(int, data[index:index+K]))

# Build the tree using adjacency list
tree = collections.defaultdict(list)
for A, B in edges:
    tree[A].append(B)
    tree[B].append(A)

# Function to find the LCA of two nodes in a tree
def find_lca(u, v, parent, depth):
    if depth[u] < depth[v]:
        u, v = v, u
    while depth[u] > depth[v]:
        u = parent[u]
    while u != v:
        u = parent[u]
        v = parent[v]
    return u

# BFS to find parent and depth of each node
def bfs(root):
    parent = {root: None}
    depth = {root: 0}
    queue = collections.deque([root])
    while queue:
        node = queue.popleft()
        for neighbor in tree[node]:
            if neighbor not in parent:
                parent[neighbor] = node
                depth[neighbor] = depth[node] + 1
                queue.append(neighbor)
    return parent, depth

# Choose an arbitrary root (1 in this case)
root = 1
parent, depth = bfs(root)

# Find the LCA of all nodes in V
lca = V[0]
for i in range(1, K):
    lca = find_lca(lca, V[i], parent, depth)

# Calculate the number of vertices in the subtree rooted at the LCA
def count_subtree_nodes(node, parent):
    count = 1
    for neighbor in tree[node]:
        if neighbor != parent:
            count += count_subtree_nodes(neighbor, node)
    return count

# Count the number of vertices in the subtree rooted at the LCA
result = count_subtree_nodes(lca, None)
print(result)