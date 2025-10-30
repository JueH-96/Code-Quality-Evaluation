import heapq

def solve():
    n, m, q = map(int, input().split())
    edges = []
    for i in range(m):
        u, v, w = map(int, input().split())
        edges.append((u, v, w))

    closed_edges = [False] * m

    for _ in range(q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            edge_index = query[1] - 1
            closed_edges[edge_index] = True
        else:
            start_node, end_node = query[1], query[2]
            
            graph = {i: [] for i in range(1, n + 1)}
            for i, (u, v, w) in enumerate(edges):
                if not closed_edges[i]:
                    graph[u].append((v, w))
                    graph[v].append((u, w))

            distances = {i: float('inf') for i in range(1, n + 1)}
            distances[start_node] = 0
            priority_queue = [(0, start_node)]

            while priority_queue:
                dist, current_node = heapq.heappop(priority_queue)

                if dist > distances[current_node]:
                    continue

                for neighbor, weight in graph[current_node]:
                    new_dist = dist + weight
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        heapq.heappush(priority_queue, (new_dist, neighbor))

            if distances[end_node] == float('inf'):
                print(-1)
            else:
                print(distances[end_node])

solve()