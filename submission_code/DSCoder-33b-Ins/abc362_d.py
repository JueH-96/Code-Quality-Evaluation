import sys
import heapq

def dijkstra(graph, start):
    dist = [float('inf')] * len(graph)
    dist[start] = 0
    queue = [(0, start)]
    while queue:
        path_len, v = heapq.heappop(queue)
        if path_len == dist[v]:
            for w, edge_len in graph[v]:
                if edge_len + path_len < dist[w]:
                    dist[w] = edge_len + path_len
                    heapq.heappush(queue, (edge_len + path_len, w))
    return dist

lines = sys.stdin.readlines()
N, M = map(int, lines[0].split())
A = list(map(int, lines[1].split()))
graph = [[] for _ in range(N)]
for line in lines[2:]:
    U, V, B = map(int, line.split())
    U -= 1
    V -= 1
    graph[U].append((V, B + A[V]))
    graph[V].append((U, B + A[U]))

dist = dijkstra(graph, 0)
for i in range(1, N):
    print(dist[i] + A[0], end=' ')