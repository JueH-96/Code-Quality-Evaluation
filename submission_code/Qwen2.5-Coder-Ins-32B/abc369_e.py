# YOUR CODE HERE
import sys
import heapq

def dijkstra(graph, start):
    n = len(graph)
    dist = [float('inf')] * n
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        current_dist, u = heapq.heappop(pq)
        if current_dist > dist[u]:
            continue
        for v, weight in graph[u]:
            distance = current_dist + weight
            if distance < dist[v]:
                dist[v] = distance
                heapq.heappush(pq, (distance, v))
    return dist

def solve():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    M = int(input[idx + 1])
    idx += 2
    bridges = []
    for _ in range(M):
        U = int(input[idx]) - 1
        V = int(input[idx + 1]) - 1
        T = int(input[idx + 2])
        bridges.append((U, V, T))
        idx += 3
    Q = int(input[idx])
    idx += 1
    queries = []
    for _ in range(Q):
        K = int(input[idx])
        idx += 1
        query_bridges = [int(input[idx + i]) - 1 for i in range(K)]
        queries.append(query_bridges)
        idx += K

    graph = [[] for _ in range(N)]
    for u, v, t in bridges:
        graph[u].append((v, t))
        graph[v].append((u, t))

    full_dist = dijkstra(graph, 0)

    results = []
    for query in queries:
        required_bridges = set(query)
        new_graph = [[] for _ in range(N)]
        for i, (u, v, t) in enumerate(bridges):
            if i in required_bridges:
                new_graph[u].append((v, t))
                new_graph[v].append((u, t))
            else:
                new_graph[u].append((v, 0))
                new_graph[v].append((u, 0))

        dist = dijkstra(new_graph, 0)
        min_time = float('inf')
        for i in required_bridges:
            u, v, t = bridges[i]
            min_time = min(min_time, dist[u] + t + full_dist[v], dist[v] + t + full_dist[u])
        results.append(min_time)

    for result in results:
        print(result)

if __name__ == "__main__":
    solve()