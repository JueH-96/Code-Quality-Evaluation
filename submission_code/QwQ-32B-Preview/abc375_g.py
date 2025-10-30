import sys
import heapq
from math import inf

def dijkstra(start, graph):
    N = len(graph)
    dist = [inf] * N
    dist[start] = 0
    heap = [(0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, c in graph[u]:
            if dist[u] + c < dist[v]:
                dist[v] = dist[u] + c
                heapq.heappush(heap, (dist[v], v))
    return dist

def main():
    import sys
    from math import inf
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    M = int(input[idx+1])
    idx += 2
    graph = [[] for _ in range(N)]
    roads = []
    for _ in range(M):
        A = int(input[idx]) - 1
        B = int(input[idx+1]) - 1
        C = int(input[idx+2])
        graph[A].append((B, C))
        graph[B].append((A, C))
        roads.append((A, B, C))
        idx += 3
    dist_start = dijkstra(0, graph)
    dist_end = dijkstra(N-1, graph)
    D_all = dist_start[N-1]
    for road in roads:
        A, B, C = road
        if dist_start[A] + C + dist_end[B] == D_all or dist_start[B] + C + dist_end[A] == D_all:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()