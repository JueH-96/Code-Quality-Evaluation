import sys
import heapq

input = sys.stdin.read
def solve():
    data = input().split()
    index = 0
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1
    
    edges = []
    graph = [[] for _ in range(N + 1)]
    
    for i in range(M):
        A = int(data[index])
        index += 1
        B = int(data[index])
        index += 1
        C = int(data[index])
        index += 1
        edges.append((A, B, C))
        graph[A].append((B, C, i))
        graph[B].append((A, C, i))
    
    def dijkstra(source, target, skip_edge=None):
        # skip_edge is the index of the edge to skip
        INF = float('inf')
        dist = [INF] * (N + 1)
        dist[source] = 0
        heap = [(0, source)]
        
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v, cost, edge_index in graph[u]:
                if edge_index == skip_edge:
                    continue
                if dist[u] + cost < dist[v]:
                    dist[v] = dist[u] + cost
                    heapq.heappush(heap, (dist[v], v))
        
        return dist[target]
    
    # Calculate the shortest path from 1 to N with all roads passable
    shortest_path_all = dijkstra(1, N)
    
    results = []
    for i in range(M):
        A, B, C = edges[i]
        # Calculate the shortest path from 1 to N skipping the i-th road
        shortest_path_skip = dijkstra(1, N, i)
        if shortest_path_skip != shortest_path_all:
            results.append("Yes")
        else:
            results.append("No")
    
    print("
".join(results))