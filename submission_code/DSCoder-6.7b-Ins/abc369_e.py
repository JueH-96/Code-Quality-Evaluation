import heapq
import sys
from collections import defaultdict

def read_input():
    N, M = map(int, sys.stdin.readline().split())
    bridges = defaultdict(list)
    for _ in range(M):
        U, V, T = map(int, sys.stdin.readline().split())
        bridges[U].append((V, T))
        bridges[V].append((U, T))
    Q = int(sys.stdin.readline())
    queries = []
    for _ in range(Q):
        K = int(sys.stdin.readline())
        B = list(map(int, sys.stdin.readline().split()))
        queries.append((K, B))
    return N, M, bridges, Q, queries

def solve():
    N, M, bridges, Q, queries = read_input()
    for K, B in queries:
        graph = [[] for _ in range(N+1)]
        for u, v, t in bridges.items():
            for w, T in v:
                graph[u].append((w, T))
                graph[w].append((u, T))
        dist = [float('inf')] * (N+1)
        dist[1] = 0
        visited = [False] * (N+1)
        heap = [(0, 1)]
        while heap:
            d, u = heapq.heappop()
            if visited[u]: continue
            visited[u] = True
            for v, T in graph[u]:
                if not visited[v] and dist[u] + T < dist[v]:
                    dist[v] = dist[u] + T
                    heapq.heappush(heap, (dist[v], v))
        print(dist[N])

solve()