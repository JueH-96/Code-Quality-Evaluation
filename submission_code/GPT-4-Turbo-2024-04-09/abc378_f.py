import sys
from collections import defaultdict, deque

input = sys.stdin.read
data = input().split()

N = int(data[0])
edges = [(int(data[i*2+1]), int(data[i*2+2])) for i in range(N-1)]

# Adjacency list for the tree
tree = defaultdict(list)
for u, v in edges:
    tree[u].append(v)
    tree[v].append(u)

# To find all pairs (u, v) that can form a cycle with all vertices having degree 3,
# we need to:
# 1. Find the shortest path between any two nodes u and v such that adding (u, v)
#    would form a cycle where all vertices in the cycle have degree 3.

# We can use BFS from each node to determine the shortest path to all other nodes
# and check the condition for the degrees.

def bfs(start):
    queue = deque([start])
    distances = {start: 0}
    parents = {start: None}
    while queue:
        node = queue.popleft()
        for neighbor in tree[node]:
            if neighbor not in distances:
                parents[neighbor] = node
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances, parents

# We need to count valid cycles
valid_cycles = 0

# Degree of each node
degree = defaultdict(int)
for u, v in edges:
    degree[u] += 1
    degree[v] += 1

# Check each pair of nodes if they can form a valid cycle
for u in range(1, N+1):
    distances, parents = bfs(u)
    for v in range(u+1, N+1):
        if v in distances:
            # Check if adding edge u-v forms a valid cycle
            path_length = distances[v]
            if path_length >= 3:  # A cycle must have at least 3 edges
                # Trace the path and check the degrees
                path_nodes = set()
                current = v
                while current != u:
                    path_nodes.add(current)
                    current = parents[current]
                path_nodes.add(u)
                
                # Check if all nodes on the path can have degree 3
                if all(degree[node] == 2 for node in path_nodes):
                    valid_cycles += 1

print(valid_cycles)