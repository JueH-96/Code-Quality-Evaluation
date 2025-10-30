def main():
    import sys, heapq
    input_data = sys.stdin.buffer.read().split()
    it = iter(input_data)
    n = int(next(it))
    m = int(next(it))
    
    # Read vertex weights (1-indexed)
    A = [0] * (n + 1)
    for i in range(1, n + 1):
        A[i] = int(next(it))
    
    # Build the graph in 1-indexed format.
    # Each edge (u, v) with weight b is represented as two directed edges.
    graph = [[] for _ in range(n + 1)]
    for _ in range(m):
        u = int(next(it))
        v = int(next(it))
        b = int(next(it))
        graph[u].append((v, b))
        graph[v].append((u, b))
    
    # We will perform Dijkstra's algorithm.
    # We define the cost to reach a vertex v as the sum of the vertex weights (each time a vertex is visited on the path)
    # and the edge weights. Initially, starting at vertex 1 we incur the cost A[1].
    
    INF = 10**20
    dist = [INF] * (n + 1)
    dist[1] = A[1]
    heap = [(A[1], 1)]
    
    while heap:
        cost, u = heapq.heappop(heap)
        if cost != dist[u]:
            continue
        for v, b in graph[u]:
            # When moving from u to v, add the edge weight b and the vertex weight A[v].
            new_cost = cost + b + A[v]
            if new_cost < dist[v]:
                dist[v] = new_cost
                heapq.heappush(heap, (new_cost, v))
    
    # Print results for vertices 2 to n in one line separated by spaces.
    result = " ".join(str(dist[i]) for i in range(2, n + 1))
    sys.stdout.write(result)

if __name__ == '__main__':
    main()