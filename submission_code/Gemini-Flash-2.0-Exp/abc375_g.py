import heapq

def solve():
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v, w = map(int, input().split())
        edges.append((u, v, w))

    def dijkstra(graph, start, end):
        dist = {node: float('inf') for node in graph}
        dist[start] = 0
        pq = [(0, start)]

        while pq:
            d, u = heapq.heappop(pq)

            if d > dist[u]:
                continue

            for v, weight in graph[u].items():
                if dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
                    heapq.heappush(pq, (dist[v], v))

        return dist[end]

    # Calculate shortest distance with all roads
    graph_all = {i: {} for i in range(1, n + 1)}
    for u, v, w in edges:
        graph_all[u][v] = w
        graph_all[v][u] = w
    
    shortest_dist_all = dijkstra(graph_all, 1, n)

    for i in range(m):
        # Calculate shortest distance without road i
        graph_without_i = {j: {} for j in range(1, n + 1)}
        for j in range(m):
            if i != j:
                u, v, w = edges[j]
                graph_without_i[u][v] = w
                graph_without_i[v][u] = w

        shortest_dist_without_i = dijkstra(graph_without_i, 1, n)

        if shortest_dist_all != shortest_dist_without_i:
            print("Yes")
        else:
            print("No")

solve()