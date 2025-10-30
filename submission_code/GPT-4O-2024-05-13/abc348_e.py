# YOUR CODE HERE
import sys
import collections

input = sys.stdin.read
data = input().split()

N = int(data[0])
edges = []
index = 1
for _ in range(N-1):
    A = int(data[index])
    B = int(data[index+1])
    edges.append((A, B))
    index += 2

C = list(map(int, data[index:]))

# Build the tree as an adjacency list
tree = collections.defaultdict(list)
for A, B in edges:
    tree[A].append(B)
    tree[B].append(A)

# Function to calculate the subtree sizes and initial f(1)
def dfs(node, parent):
    subtree_size = C[node-1]
    total_cost = 0
    for neighbor in tree[node]:
        if neighbor != parent:
            child_size, child_cost = dfs(neighbor, node)
            subtree_size += child_size
            total_cost += child_cost + child_size
    return subtree_size, total_cost

# Calculate initial f(1)
subtree_sizes = [0] * (N + 1)
total_costs = [0] * (N + 1)
subtree_sizes[1], total_costs[1] = dfs(1, -1)

# Function to calculate f(v) for all v using f(1)
def reroot(node, parent):
    for neighbor in tree[node]:
        if neighbor != parent:
            total_costs[neighbor] = total_costs[node] + (subtree_sizes[1] - 2 * subtree_sizes[neighbor])
            reroot(neighbor, node)

# Reroot the tree to calculate f(v) for all v
reroot(1, -1)

# Find the minimum f(v)
min_f = float('inf')
for i in range(1, N + 1):
    min_f = min(min_f, total_costs[i])

print(min_f)