from heapq import *

N, M = map(int, input().split())
edges = [[] for _ in range(N)]
for _ in range(M):
    u, v = map(int, input().split())
    u -= 1
    v -= 1
    edges[u].append(v)
    edges[v].append(u)
W = list(map(int, input().split()))
A = list(map(int, input().split()))

h = []
for i in range(N):
    heappush(h, (-W[i], i))

res = 0
while h:
    w, u = heappop(h)
    w = -w
    if A[u] == 0:
        continue
    A[u] -= 1
    res += 1
    for v in edges[u]:
        if W[v] < w:
            A[v] += 1
print(res)