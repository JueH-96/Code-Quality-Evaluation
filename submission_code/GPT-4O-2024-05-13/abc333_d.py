# YOUR CODE HERE
import sys
from collections import deque, defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
edges = [(int(data[2*i+1]), int(data[2*i+2])) for i in range(N-1)]

# Create an adjacency list for the tree
adj = defaultdict(list)
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

# BFS to find the farthest node from node 1
def bfs(start):
    visited = [False] * (N + 1)
    queue = deque([(start, 0)])
    visited[start] = True
    farthest_node = start
    max_distance = 0
    
    while queue:
        node, distance = queue.popleft()
        if distance > max_distance:
            max_distance = distance
            farthest_node = node
        for neighbor in adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append((neighbor, distance + 1))
    
    return farthest_node, max_distance

# Find the farthest node from node 1
farthest_node_from_1, _ = bfs(1)
# Find the farthest node from the farthest node found above
_, max_distance = bfs(farthest_node_from_1)

# The minimum number of operations required to delete vertex 1 is the maximum distance found
print((max_distance + 1) // 2)