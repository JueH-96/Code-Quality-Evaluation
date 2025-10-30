import heapq

def solve():
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v, w = map(int, input().split())
        edges.append((u - 1, v - 1, w))

    def dijkstra(graph):
        dist = [float('inf')] * n
        dist[0] = 0
        pq = [(0, 0)]

        while pq:
            d, u = heapq.heappop(pq)

            if d > dist[u]:
                continue

            for v, w in graph.get(u, []):
                if dist[v] > dist[u] + w:
                    dist[v] = dist[u] + w
                    heapq.heappush(pq, (dist[v], v))
        return dist

    def build_graph(included_edges):
        graph = {}
        for u, v, w in included_edges:
            graph.setdefault(u, []).append((v, w))
            graph.setdefault(v, []).append((u, w))
        return graph

    all_edges_graph = build_graph(edges)
    all_edges_dist = dijkstra(all_edges_graph)
    all_shortest = all_edges_dist[n - 1]

    for i in range(m):
        excluded_edge = edges[i]
        included_edges = edges[:i] + edges[i+1:]
        
        graph = build_graph(included_edges)
        dist = dijkstra(graph)
        shortest = dist[n-1]

        if shortest != all_shortest:
            print("Yes")
        else:
            print("No")

solve()