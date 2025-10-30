# YOUR CODE HERE
import sys
import heapq

def dijkstra(N, A, B, C, D):
    INF = float('inf')
    dist = [[INF]*2 for _ in range(N)]
    dist[0][0] = 0
    pq = [(0, 0, 0)]
    while pq:
        d, u, c = heapq.heappop(pq)
        if d > dist[u][c]:
            continue
        for v in range(N):
            if v == u:
                continue
            if c == 0:
                alt = d + D[u][v]*A
                if alt < dist[v][0]:
                    dist[v][0] = alt
                    heapq.heappush(pq, (alt, v, 0))
                alt = d + D[u][v]*B + C
                if alt < dist[v][1]:
                    dist[v][1] = alt
                    heapq.heappush(pq, (alt, v, 1))
            else:
                alt = d + D[u][v]*B + C
                if alt < dist[v][1]:
                    dist[v][1] = alt
                    heapq.heappush(pq, (alt, v, 1))
    return min(dist[N-1])

N, A, B, C = map(int, sys.stdin.readline().split())
D = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]
print(dijkstra(N, A, B, C, D))