import sys
from collections import defaultdict, deque

# Read input
input = sys.stdin.read
data = input().split()

N1 = int(data[0])
N2 = int(data[1])
M = int(data[2])

edges = []
for i in range(M):
    a = int(data[3 + 2 * i]) - 1
    b = int(data[4 + 2 * i]) - 1
    edges.append((a, b))

# Create adjacency list
adj_list = defaultdict(list)
for a, b in edges:
    adj_list[a].append(b)
    adj_list[b].append(a)

# Function to find the shortest path from start to end using BFS
def bfs(start, end):
    queue = deque([start])
    visited = set([start])
    distance = {start: 0}
    
    while queue:
        current = queue.popleft()
        if current == end:
            return distance[end]
        for neighbor in adj_list[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                distance[neighbor] = distance[current] + 1
                queue.append(neighbor)
    
    return float('inf')

# Find the maximum possible d
max_d = 0
for u in range(N1):
    for v in range(N1, N1 + N2):
        if v not in adj_list[u]:
            d = bfs(u, v)
            if d > max_d:
                max_d = d

# Print the result
print(max_d)