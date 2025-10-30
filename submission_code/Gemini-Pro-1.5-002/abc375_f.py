import heapq

def solve():
    n, m, q = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v, w = map(int, input().split())
        edges.append((u - 1, v - 1, w))

    adj = [[] for _ in range(n)]
    for i in range(m):
        u, v, w = edges[i]
        adj[u].append((v, w, i))
        adj[v].append((u, w, i))
    
    closed_roads = set()

    for _ in range(q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            road_idx = query[1] - 1
            closed_roads.add(road_idx)
        else:
            start_node = query[1] - 1
            end_node = query[2] - 1

            dist = [float('inf')] * n
            dist[start_node] = 0
            pq = [(0, start_node)]

            while pq:
                d, u = heapq.heappop(pq)

                if d > dist[u]:
                    continue

                for v, w, road_idx in adj[u]:
                    if road_idx not in closed_roads:
                        if dist[u] + w < dist[v]:
                            dist[v] = dist[u] + w
                            heapq.heappush(pq, (dist[v], v))

            if dist[end_node] == float('inf'):
                print(-1)
            else:
                print(dist[end_node])

solve()