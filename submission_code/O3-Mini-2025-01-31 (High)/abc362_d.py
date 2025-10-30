def main():
    import sys, heapq
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    # Read vertex weights (0-indexed: vertex i corresponds to A[i] where i=0...n-1)
    A = [int(next(it)) for _ in range(n)]
    
    # Build an adjacency list for the undirected graph.
    graph = [[] for _ in range(n)]
    for _ in range(m):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        w = int(next(it))
        graph[u].append((v, w))
        graph[v].append((u, w))
    
    # We'll use Dijkstra's algorithm.
    # The trick is that the cost of going from u to v is:
    #   current_cost + edge_weight + A[v]
    # We initialize the starting vertex (vertex 1, index 0) with cost A[0].
    INF = float('inf')
    dist = [INF] * n
    dist[0] = A[0]
    pq = [(dist[0], 0)]  # (distance, vertex)
    
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        for v, edge_weight in graph[u]:
            new_cost = d + edge_weight + A[v]
            if new_cost < dist[v]:
                dist[v] = new_cost
                heapq.heappush(pq, (new_cost, v))
                
    # Print the answers for vertices 2 through n (i.e. indices 1 through n-1)
    sys.stdout.write(" ".join(str(dist[i]) for i in range(1, n)))
    
if __name__ == '__main__':
    main()