import heapq

def dijkstra(n, edges, start, end, forbidden_edges):
    graph = [[] for _ in range(n + 1)]
    for u, v, t in edges:
        graph[u].append((v, t))
        graph[v].append((u, t))

    pq = [(0, start)]
    dist = [float('inf')] * (n + 1)
    dist[start] = 0

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, t in graph[u]:
            if (u, v) in forbidden_edges or (v, u) in forbidden_edges:
                continue
            if dist[v] > dist[u] + t:
                dist[v] = dist[u] + t
                heapq.heappush(pq, (dist[v], v))

    return dist[end]

def main():
    import sys
    input = sys.stdin.read
    data = input().split()

    index = 0
    N = int(data[index])
    M = int(data[index + 1])
    index += 2

    edges = []
    for _ in range(M):
        U = int(data[index])
        V = int(data[index + 1])
        T = int(data[index + 2])
        edges.append((U, V, T))
        index += 3

    Q = int(data[index])
    index += 1

    results = []
    for _ in range(Q):
        K = int(data[index])
        index += 1
        bridges = set()
        for _ in range(K):
            B = int(data[index])
            index += 1
            bridges.add(B)

        forbidden_edges = set()
        for i in range(M):
            if i + 1 not in bridges:
                forbidden_edges.add((edges[i][0], edges[i][1]))

        min_time = dijkstra(N, edges, 1, N, forbidden_edges)
        results.append(min_time)

    for result in results:
        print(result)

if __name__ == "__main__":
    main()