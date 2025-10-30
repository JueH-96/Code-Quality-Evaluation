import heapq

def dijkstra(graph, start, end, closed_roads):
    n = len(graph)
    dist = [float('inf')] * n
    dist[start] = 0
    pq = [(0, start)]

    while pq:
        current_dist, u = heapq.heappop(pq)

        if current_dist > dist[u]:
            continue

        for v, weight in graph[u]:
            if (u, v) in closed_roads or (v, u) in closed_roads:
                continue
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                heapq.heappush(pq, (dist[v], v))

    return dist[end] if dist[end] != float('inf') else -1

def main():
    import sys
    input = sys.stdin.read
    data = input().split()

    index = 0
    N = int(data[index])
    M = int(data[index + 1])
    Q = int(data[index + 2])
    index += 3

    graph = [[] for _ in range(N + 1)]
    for _ in range(M):
        A = int(data[index])
        B = int(data[index + 1])
        C = int(data[index + 2])
        graph[A].append((B, C))
        graph[B].append((A, C))
        index += 3

    closed_roads = set()

    for _ in range(Q):
        query_type = int(data[index])
        if query_type == 1:
            i = int(data[index + 1])
            A = int(data[3 * i - 2])
            B = int(data[3 * i - 1])
            closed_roads.add((A, B))
            index += 2
        else:
            x = int(data[index + 1])
            y = int(data[index + 2])
            result = dijkstra(graph, x, y, closed_roads)
            print(result)
            index += 3

if __name__ == "__main__":
    main()