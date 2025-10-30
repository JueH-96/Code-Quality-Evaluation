import sys
from collections import defaultdict, deque

input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
M = int(data[index + 1])
index += 2

sets = []
for _ in range(N):
    A_i = int(data[index])
    index += 1
    S_i = set(data[index:index + A_i])
    index += A_i
    sets.append(S_i)

# Create a graph where each node represents a set and an edge exists between two sets if they share at least one common element
graph = defaultdict(list)
for i in range(N):
    for j in range(i + 1, N):
        if sets[i] & sets[j]:
            graph[i].append(j)
            graph[j].append(i)

# BFS to find the shortest path from a set containing 1 to a set containing M
visited = [False] * N
queue = deque()

# Initialize the queue with all sets containing 1
for i in range(N):
    if '1' in sets[i]:
        queue.append((i, 0))
        visited[i] = True

# Perform BFS
while queue:
    current, steps = queue.popleft()
    if '1' in sets[current] and str(M) in sets[current]:
        print(steps)
        break
    for neighbor in graph[current]:
        if not visited[neighbor]:
            visited[neighbor] = True
            queue.append((neighbor, steps + 1))
else:
    print(-1)