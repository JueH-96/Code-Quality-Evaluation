import sys
from collections import deque, defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
edges = [(int(data[i*2+1]), int(data[i*2+2])) for i in range(N-1)]

# Create adjacency list
adj = defaultdict(list)
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

# Find all leaves
degrees = defaultdict(int)
for node in adj:
    degrees[node] = len(adj[node])

leaves = deque(node for node in degrees if degrees[node] == 1)

# To store the pairs of leaves removed
result = []

# BFS to find distances from an arbitrary leaf
def bfs(start):
    queue = deque([start])
    distances = {start: 0}
    parent = {start: None}
    
    while queue:
        node = queue.popleft()
        for neighbor in adj[node]:
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                parent[neighbor] = node
                queue.append(neighbor)
    
    return distances, parent

# Find the farthest leaf from an arbitrary leaf to ensure maximum distance
distances_from_first_leaf, _ = bfs(leaves[0])
farthest_leaf = max(distances_from_first_leaf, key=distances_from_first_leaf.get)

# Now perform BFS from the farthest leaf to get the maximum distance leaf pairs
distances, parent = bfs(farthest_leaf)

# Sort leaves by distance from the farthest leaf
sorted_leaves = sorted((distances[leaf], leaf) for leaf in leaves if leaf in distances)
sorted_leaves.reverse()

# Matching leaves from opposite ends to maximize distance
while len(sorted_leaves) >= 2:
    _, leaf1 = sorted_leaves.pop(0)
    _, leaf2 = sorted_leaves.pop(0)
    result.append((leaf1, leaf2))
    # Remove these leaves and update the tree
    p1 = parent[leaf1]
    p2 = parent[leaf2]
    if p1:
        adj[p1].remove(leaf1)
        degrees[p1] -= 1
        if degrees[p1] == 1:
            sorted_leaves.append((distances[p1], p1))
    if p2:
        adj[p2].remove(leaf2)
        degrees[p2] -= 1
        if degrees[p2] == 1:
            sorted_leaves.append((distances[p2], p2))
    sorted_leaves.sort(reverse=True, key=lambda x: x[0])

# Output the result
for x, y in result:
    print(x, y)