import sys
from collections import deque

input = sys.stdin.read
data = input().split()

N = int(data[0])
index = 1
dependencies = [[] for _ in range(N + 1)]
indegree = [0] * (N + 1)

for i in range(1, N + 1):
    C_i = int(data[index])
    index += 1
    for _ in range(C_i):
        P_ij = int(data[index])
        dependencies[P_ij].append(i)
        indegree[i] += 1
        index += 1

# Topological sort using Kahn's algorithm
queue = deque([1])
result = []

while queue:
    current = queue.popleft()
    for neighbor in dependencies[current]:
        indegree[neighbor] -= 1
        if indegree[neighbor] == 0:
            queue.append(neighbor)
            result.append(neighbor)

print(" ".join(map(str, result)))