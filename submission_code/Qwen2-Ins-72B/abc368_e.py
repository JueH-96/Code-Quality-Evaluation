from heapq import heappush, heappop

N, M, X1 = map(int, input().split())
ABST = [list(map(int, input().split())) for _ in range(M)]

G = [[] for _ in range(M)]
for i, (A, B, S, T) in enumerate(ABST):
    A -= 1
    B -= 1
    G[i].append((A, S))
    G[i].append((B, T))

INF = 10**18
dist = [INF] * M
dist[0] = X1
h = [(X1, 0)]
while h:
    x, v = heappop(h)
    if dist[v] < x:
        continue
    for u, t in G[v]:
        if t + x < dist[u]:
            dist[u] = t + x
            heappush(h, (dist[u], u))

X = [0] * M
for i, (A, B, S, T) in enumerate(ABST):
    A -= 1
    B -= 1
    if T + dist[i] > S + dist[i]:
        X[i] = T + dist[i] - S - dist[i]

print(*X[1:])