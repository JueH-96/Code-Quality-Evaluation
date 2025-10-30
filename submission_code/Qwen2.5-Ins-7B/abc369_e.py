# YOUR CODE HERE
from collections import defaultdict
from heapq import heappop, heappush

def dijkstra(n, graph, start, bridges):
    dist = [float('inf')] * n
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, u = heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                heappush(pq, (dist[v], v))
    return dist

def solve(n, m, edges, q, queries):
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u-1].append((v-1, w))
        graph[v-1].append((u-1, w))
    
    for i, (k, bridges) in enumerate(queries):
        queries[i] = (k, bridges, dijkstra(n, graph, 0, bridges))
    
    for k, bridges, dist in queries:
        min_time = float('inf')
        for mask in range(1 << k):
            time = 0
            for i in range(k):
                if mask & (1 << i):
                    time += edges[bridges[i] - 1][2]
            time += dist[n-1]
            min_time = min(min_time, time)
        print(min_time)

# Read input
n, m = map(int, input().split())
edges = [tuple(map(int, input().split())) for _ in range(m)]
q = int(input())
queries = []
for _ in range(q):
    k = int(input())
    bridges = list(map(int, input().split()))
    queries.append((k, bridges))

# Solve the problem
solve(n, m, edges, q, queries)