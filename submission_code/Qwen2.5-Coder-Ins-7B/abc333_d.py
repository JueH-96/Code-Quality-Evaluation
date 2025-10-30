import sys
from collections import deque

# Read input
input = sys.stdin.read
data = input().split()

N = int(data[0])
edges = []

for i in range(N - 1):
    u = int(data[2 * i + 1])
    v = int(data[2 * i + 2])
    edges.append((u, v))

# Build the tree
tree = [[] for _ in range(N + 1)]

for u, v in edges:
    tree[u].append(v)
    tree[v].append(u)

# Find the leaf nodes
leaves = [i for i in range(1, N + 1) if len(tree[i]) == 1]

# Perform the operations
operations = 0
while leaves:
    new_leaves = []
    for leaf in leaves:
        parent = tree[leaf][0]
        tree[parent].remove(leaf)
        if len(tree[parent]) == 1:
            new_leaves.append(parent)
    leaves = new_leaves
    operations += 1

# Print the result
print(operations)