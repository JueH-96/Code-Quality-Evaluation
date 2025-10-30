def main():
    import sys, heapq
    input = sys.stdin.readline

    # Read number of vertices and edges
    N, M = map(int, input().split())
    
    # Read vertex weights. Convert index offset (0-indexed)
    A = list(map(int, input().split()))
    
    # Build graph as an adjacency list.
    # Each entry is a tuple (neighbor, edge_weight)
    graph = [[] for _ in range(N)]
    for _ in range(M):
        u, v, b = map(int, input().split())
        # Convert vertices to 0-index and store both ways (undirected)
        u -= 1
        v -= 1
        graph[u].append((v, b))
        graph[v].append((u, b))
    
    # Use Dijkstra's algorithm from vertex 1 (index 0).
    # Initialize distances; we add weight of starting vertex A[0] to the starting cost.
    INF = 10**18  # Set a sufficiently high value (can use float('inf') as well)
    dist = [INF] * N
    dist[0] = A[0]
    
    # Priority queue with (distance, vertex)
    hq = [(dist[0], 0)]
    
    while hq:
        d, u = heapq.heappop(hq)
        if d != dist[u]:
            continue
        # Traverse all neighbors
        for v, edge_weight in graph[u]:
            # The cost to travel from u -> v is current distance + edge cost + vertex v weight.
            nd = d + edge_weight + A[v]
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(hq, (nd, v))
    
    # Output the results for vertices 2 to N (index 1 to N-1)
    result = " ".join(str(dist[i]) for i in range(1, N))
    sys.stdout.write(result)

if __name__ == '__main__':
    main()