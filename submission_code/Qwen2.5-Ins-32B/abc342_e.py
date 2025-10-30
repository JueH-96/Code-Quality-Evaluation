# YOUR CODE HERE
import sys
from heapq import heappush, heappop

def dijkstra(N, graph):
    dist = [-float('inf')] * N
    dist[N-1] = 0
    pq = [(0, N-1)]
    while pq:
        d, u = heappop(pq)
        d = -d
        if dist[u] > d:
            continue
        for v, w in graph[u]:
            if dist[v] < d + w:
                dist[v] = d + w
                heappush(pq, (-dist[v], v))
    return dist

def main():
    input = sys.stdin.read
    N, M, *rest = map(int, input().split())
    graph = [[] for _ in range(N)]
    idx = 0
    for _ in range(M):
        l, d, k, c, A, B = rest[idx:idx+6]
        idx += 6
        A -= 1
        B -= 1
        for i in range(k):
            t = l + i * d
            graph[B].append((A, t - c))
            graph[A].append((B, -t - c))
    
    dist = dijkstra(N, graph)
    for i in range(N-1):
        if dist[i] == -float('inf'):
            print("Unreachable")
        else:
            print(-dist[i])

main()