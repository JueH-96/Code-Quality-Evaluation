from heapq import heappush, heappop

N, M = map(int, input().split())
edges = [[] for _ in range(N)]
for _ in range(M):
    u, v, b, c = map(int, input().split())
    edges[u-1].append((v-1, b, c))

def dijkstra(start, goal, edges, N):
    INF = float('inf')
    dist = [INF] * N
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, v = heappop(pq)
        if d > dist[v]:
            continue
        for u, b, c in edges[v]:
            if dist[v] + c < dist[u]:
                dist[u] = dist[v] + c
                heappush(pq, (dist[u], u))
    return dist

cost = dijkstra(0, N-1, edges, N)
if cost[N-1] == float('inf'):
    print(0)
    return

def max_ratio_path(start, goal, edges, N):
    INF = float('inf')
    ratio = [0] * N
    ratio[start] = INF
    pq = [(INF, start)]
    while pq:
        r, v = heappop(pq)
        if r < ratio[v]:
            continue
        for u, b, c in edges[v]:
            if cost[u] == INF or cost[u] - cost[v] > c:
                continue
            new_ratio = (ratio[v] * cost[v] + b) / (cost[v] + c)
            if new_ratio > ratio[u]:
                ratio[u] = new_ratio
                heappush(pq, (ratio[u], u))
    return ratio[goal]

print(max_ratio_path(0, N-1, edges, N))