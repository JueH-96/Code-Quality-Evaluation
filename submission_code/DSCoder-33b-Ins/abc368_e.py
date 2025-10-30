# YOUR CODE HERE
import sys
from heapq import heappop, heappush

def solve():
    read = sys.stdin.readline
    N, M, X1 = map(int, read().split())
    graph = [[] for _ in range(N)]
    for _ in range(M):
        A, B, S, T = map(int, read().split())
        A -= 1
        B -= 1
        graph[A].append((B, S, T))

    def dijkstra(start):
        dist = [float('inf')] * N
        dist[start] = X1
        heap = [(X1, start)]
        while heap:
            d, node = heappop(heap)
            if dist[node] < d:
                continue
            for nei, S, T in graph[node]:
                if d <= S:
                    if dist[nei] > T:
                        dist[nei] = T
                        heappush(heap, (T, nei))
                else:
                    if dist[nei] > d:
                        dist[nei] = d
                        heappush(heap, (d, nei))
        return dist

    dists = [dijkstra(i) for i in range(N)]
    ans = [0] * M
    for i in range(M):
        A, B, S, T = map(int, read().split())
        A -= 1
        B -= 1
        ans[i] = dists[A][B] - T
    print(*ans)

solve()