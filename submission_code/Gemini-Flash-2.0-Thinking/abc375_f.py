import heapq

def solve():
    n, m, q = map(int, input().split())
    edges = []
    for i in range(m):
        u, v, c = map(int, input().split())
        edges.append((u - 1, v - 1, c))

    closed_roads = set()

    def dijkstra(start, end, current_edges, closed):
        dist = {i: float('inf') for i in range(n)}
        dist[start] = 0
        pq = [(0, start)]

        while pq:
            d, u = heapq.heappop(pq)

            if d > dist[u]:
                continue

            for i, (v1, v2, weight) in enumerate(current_edges):
                neighbor = -1
                if v1 == u and i not in closed:
                    neighbor = v2
                elif v2 == u and i not in closed:
                    neighbor = v1

                if neighbor != -1:
                    if dist[u] + weight < dist[neighbor]:
                        dist[neighbor] = dist[u] + weight
                        heapq.heappush(pq, (dist[neighbor], neighbor))

        return dist[end] if dist[end] != float('inf') else -1

    for _ in range(q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            closed_roads.add(query[1] - 1)
        else:
            x, y = query[1] - 1, query[2] - 1
            result = dijkstra(x, y, edges, closed_roads)
            print(result)

solve()