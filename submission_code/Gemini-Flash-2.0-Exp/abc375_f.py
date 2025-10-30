import heapq

def solve():
    n, m, q = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v, w = map(int, input().split())
        edges.append((u, v, w))

    closed_roads = set()

    for _ in range(q):
        query = list(map(int, input().split()))
        
        if query[0] == 1:
            closed_roads.add(query[1] - 1)
        else:
            start_node = query[1]
            end_node = query[2]

            graph = {i: [] for i in range(1, n + 1)}
            for i in range(m):
                if i not in closed_roads:
                    u, v, w = edges[i]
                    graph[u].append((v, w))
                    graph[v].append((u, w))

            distances = {i: float('inf') for i in range(1, n + 1)}
            distances[start_node] = 0
            pq = [(0, start_node)]

            while pq:
                dist, u = heapq.heappop(pq)

                if dist > distances[u]:
                    continue

                for v, weight in graph[u]:
                    if distances[v] > distances[u] + weight:
                        distances[v] = distances[u] + weight
                        heapq.heappush(pq, (distances[v], v))

            if distances[end_node] == float('inf'):
                print(-1)
            else:
                print(distances[end_node])

solve()