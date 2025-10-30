from collections import deque

INF = float('inf')

def dijkstra(vec, start):
    inf = float("inf")
    h = deque([start])
    dist = [inf] * len(vec)
    dist[start] = 0
    seen = [False] * len(vec)

    while h:
        node = h.popleft()
        seen[node] = True

        for dest, weight in vec[node]:
            if seen[dest]: continue

            if dist[dest] > dist[node] + weight:
                dist[dest] = dist[node] + weight
                h.append(dest)

    return dist


N, M, Q = map(int, input().split())

grafo = [deque([]) for _ in range(N+1)]

for _ in range(M):
    A, B, C = map(int, input().split())
    grafo[A].append((B, C))
    grafo[B].append((A, C))
cerrados = set()

for _ in range(Q):
    x, y, *rest = map(int, input().split())
    if x == 1:
        cerrados.add(y)
    elif x == 2:
        D = INF
        for next in range(1, N+1):
            for dest, ID in grafo[next]:
                if y in cerrados: continue
                if ID == y: continue
                D = min(D, dijkstra(grafo, next)[y])
        if D == INF or D == float('inf'):
            print(-1)
        else:
            print(D)
    else:
        assert False