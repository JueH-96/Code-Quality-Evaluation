import heapq

def dijkstra(n, edges, source, blocked_edge):
    # Create a priority queue to store (distance, node)
    pq = [(0, source)]
    # Create a distance array with infinity values
    dist = [float('inf')] * (n + 1)
    dist[source] = 0

    while pq:
        current_dist, u = heapq.heappop(pq)

        if current_dist > dist[u]:
            continue

        for v, w, edge_id in edges[u]:
            if edge_id == blocked_edge:
                continue
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))

    return dist

def main():
    import sys
    input = sys.stdin.read
    data = input().split()

    index = 0
    n = int(data[index])
    m = int(data[index + 1])
    index += 2

    edges = [[] for _ in range(n + 1)]
    for i in range(m):
        a = int(data[index])
        b = int(data[index + 1])
        c = int(data[index + 2])
        index += 3
        edges[a].append((b, c, i))
        edges[b].append((a, c, i))

    # Calculate shortest distances from city 1 to all other cities when all roads are passable
    dist_all = dijkstra(n, edges, 1, -1)

    result = []
    for i in range(m):
        # Calculate shortest distances from city 1 to all other cities when road i is blocked
        dist_blocked = dijkstra(n, edges, 1, i)
        if dist_all[n] != dist_blocked[n]:
            result.append("Yes")
        else:
            result.append("No")

    print("
".join(result))

if __name__ == "__main__":
    main()