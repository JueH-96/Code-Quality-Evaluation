import heapq

def solve():
    n, m = map(int, input().split())
    edges = []
    for i in range(m):
        u, v, w = map(int, input().split())
        edges.append((u, v, w))

    def dijkstra(graph):
        dist = {i: float('inf') for i in range(1, n + 1)}
        dist[1] = 0
        pq = [(0, 1)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w in graph.get(u, []):
                if dist[v] > dist[u] + w:
                    dist[v] = dist[u] + w
                    heapq.heappush(pq, (dist[v], v))
        return dist[n] if dist[n] != float('inf') else float('inf')

    full_graph = {}
    for u, v, w in edges:
        full_graph.setdefault(u, []).append((v, w))
        full_graph.setdefault(v, []).append((u, w))
    
    full_dist = dijkstra(full_graph)

    for i in range(m):
        temp_graph = {}
        for j in range(m):
            if i != j:
                u, v, w = edges[j]
                temp_graph.setdefault(u, []).append((v, w))
                temp_graph.setdefault(v, []).append((u, w))
        
        temp_dist = dijkstra(temp_graph)
        
        if (full_dist == float('inf') and temp_dist != float('inf')) or \
           (full_dist != float('inf') and temp_dist == float('inf')) or \
           (full_dist != float('inf') and temp_dist != float('inf') and full_dist != temp_dist):
            print("Yes")
        else:
            print("No")

solve()