import sys
from collections import deque

input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
X = int(data[index + 1])
index += 2
A = list(map(int, data[index:index + N]))
index += N
B = list(map(int, data[index:index + N]))
index += N
P = list(map(int, data[index:index + N]))
index += N
Q = list(map(int, data[index:index + N]))

# Create a graph where each node represents a box
# and there is an edge from box i to box P[i] and Q[i]
graph = [[] for _ in range(N + 1)]
for i in range(N):
    graph[P[i]].append(i + 1)
    graph[Q[i]].append(i + 1)

# Perform a BFS to find the shortest path from each box to the target box X
queue = deque([X])
dist = [-1] * (N + 1)
dist[X] = 0

while queue:
    u = queue.popleft()
    for v in graph[u]:
        if dist[v] == -1:
            dist[v] = dist[u] + 1
            queue.append(v)

# Check if all boxes can be emptied except the target box X
can_empty = all(dist[i] != -1 for i in range(1, N + 1) if i != X)

if not can_empty:
    print(-1)
else:
    # Calculate the minimum number of operations needed
    min_operations = sum(A) + sum(B) - (A[X - 1] + B[X - 1])
    print(min_operations)