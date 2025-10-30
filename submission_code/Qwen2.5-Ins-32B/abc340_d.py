import sys
from heapq import heappop, heappush

def solve():
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    A = [0] * (n - 1)
    B = [0] * (n - 1)
    X = [0] * (n - 1)
    idx = 1
    for i in range(n - 1):
        A[i] = int(data[idx])
        B[i] = int(data[idx + 1])
        X[i] = int(data[idx + 2]) - 1
        idx += 3

    # Dijkstra's algorithm
    INF = float('inf')
    dist = [INF] * n
    dist[0] = 0
    pq = [(0, 0)]  # (distance, node)
    while pq:
        d, u = heappop(pq)
        if d != dist[u]:
            continue
        for v in range(u + 1, n):
            if dist[v] > dist[u] + A[u]:
                dist[v] = dist[u] + A[u]
                heappush(pq, (dist[v], v))
        if dist[X[u]] > dist[u] + B[u]:
            dist[X[u]] = dist[u] + B[u]
            heappush(pq, (dist[X[u]], X[u]))

    print(dist[n - 1])

solve()