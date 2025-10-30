import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
index = 2

sets = []
for i in range(N):
    A_i = int(data[index])
    S_i = list(map(int, data[index+1:index+1+A_i]))
    sets.append(S_i)
    index += 1 + A_i

# Check if 1 and M are already in the same set
for s in sets:
    if 1 in s and M in s:
        print(0)
        sys.exit(0)

# Check if 1 and M can be obtained by merging sets
from collections import defaultdict

# Create a graph where each node is a set and there is an edge between two nodes if they have a common element
graph = defaultdict(list)
for i in range(N):
    for j in range(i+1, N):
        if any(x in sets[j] for x in sets[i]):
            graph[i].append(j)
            graph[j].append(i)

# Perform BFS to find the minimum number of operations required to merge sets containing 1 and M
from collections import deque

queue = deque()
visited = [False] * N
for i in range(N):
    if 1 in sets[i]:
        queue.append((i, 0))
        visited[i] = True

while queue:
    u, dist = queue.popleft()
    if M in sets[u]:
        print(dist)
        sys.exit(0)
    for v in graph[u]:
        if not visited[v]:
            visited[v] = True
            queue.append((v, dist + 1))

print(-1)