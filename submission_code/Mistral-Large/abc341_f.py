import sys
from collections import deque

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

edges = [[] for _ in range(N)]
for i in range(M):
    u = int(data[2 * i + 2]) - 1
    v = int(data[2 * i + 3]) - 1
    edges[u].append(v)
    edges[v].append(u)

W = [int(data[2 * M + 2 + i]) for i in range(N)]
A = [int(data[2 * M + 2 + N + i]) for i in range(N)]

def bfs(start):
    dist = [-1] * N
    dist[start] = 0
    q = deque([start])
    while q:
        x = q.popleft()
        for y in edges[x]:
            if dist[y] == -1 and W[y] < W[start]:
                dist[y] = dist[x] + 1
                q.append(y)
    return dist

max_operations = 0
for i in range(N):
    if A[i] > 0:
        dist = bfs(i)
        for j in range(N):
            if dist[j] != -1:
                max_operations = max(max_operations, dist[j])

total_operations = sum(A) + max_operations
print(total_operations)