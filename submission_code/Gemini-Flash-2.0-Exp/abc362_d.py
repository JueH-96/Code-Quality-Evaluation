import heapq

def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    edges = []
    for _ in range(m):
        u, v, b = map(int, input().split())
        edges.append((u - 1, v - 1, b))

    def dijkstra(start_node):
        dist = [float('inf')] * n
        dist[start_node] = a[start_node]
        pq = [(dist[start_node], start_node)]

        while pq:
            d, u = heapq.heappop(pq)

            if d > dist[u]:
                continue

            for v1, v2, weight in edges:
                if v1 == u:
                    v = v2
                    w = weight
                elif v2 == u:
                    v = v1
                    w = weight
                else:
                    continue

                if dist[v] > dist[u] + w + a[v]:
                    dist[v] = dist[u] + w + a[v]
                    heapq.heappush(pq, (dist[v], v))
        return dist

    distances = dijkstra(0)
    
    result = []
    for i in range(1, n):
        result.append(str(distances[i]))
    print(" ".join(result))

solve()