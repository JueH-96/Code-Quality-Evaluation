import sys
from collections import deque

# Read input
input = sys.stdin.read
data = input().split()

# Parse input
N = int(data[0])
M = int(data[1])
K = int(data[2])

edges = []
for i in range(M):
    a = int(data[3 + 2 * i]) - 1
    b = int(data[4 + 2 * i]) - 1
    edges.append((a, b))

guards = []
for i in range(K):
    p = int(data[3 + 2 * M + 2 * i]) - 1
    h = int(data[4 + 2 * M + 2 * i])
    guards.append((p, h))

# Initialize adjacency list
adj_list = [[] for _ in range(N)]
for a, b in edges:
    adj_list[a].append(b)
    adj_list[b].append(a)

# Initialize visited array
visited = [False] * N

# Function to perform BFS and mark guarded vertices
def bfs(start, h):
    queue = deque([start])
    visited[start] = True
    while queue:
        current = queue.popleft()
        for neighbor in adj_list[current]:
            if not visited[neighbor]:
                if h > 0:
                    visited[neighbor] = True
                    queue.append(neighbor)
                h -= 1

# Mark all guarded vertices
for p, h in guards:
    bfs(p, h)

# Collect guarded vertices
guarded_vertices = [i + 1 for i in range(N) if visited[i]]

# Print the result
print(len(guarded_vertices))
print(" ".join(map(str, guarded_vertices)))