import sys
import heapq

def solve():
    """
    This function implements the solution to the shortest path problem.
    It reads graph information from standard input, calculates the minimum
    path weights from vertex 1 to all other vertices using a modified
    Dijkstra's algorithm, and prints the results to standard output.
    """

    # Read problem parameters from standard input.
    # Using sys.stdin.readline for faster I/O with large inputs.
    try:
        line = sys.stdin.readline()
        if not line: return # Handle empty input
        N, M = map(int, line.split())
        A = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        return # Exit if input is malformed

    # Represent the graph using an adjacency list.
    # adj[u] will store a list of tuples (v, edge_weight).
    # We use 0-based indexing for vertices (0 to N-1).
    adj = [[] for _ in range(N)]
    for _ in range(M):
        try:
            line = sys.stdin.readline()
            if not line: break
            U, V, B = map(int, line.split())
        except (IOError, ValueError):
            continue
        # Convert 1-based vertex numbers to 0-based indices.
        U -= 1
        V -= 1
        # Since the graph is undirected, add the edge in both directions.
        adj[U].append((V, B))
        adj[V].append((U, B))

    # --- Dijkstra's Algorithm ---
    # The 'dist' array stores the minimum path weight from the source to each vertex.
    # Initialize all distances to infinity.
    dist = [float('inf')] * N
    
    # The priority queue stores (cost, vertex) tuples, ordered by cost.
    pq = []

    # The source vertex is 1, which corresponds to index 0.
    source = 0
    
    # The weight of a path from the source to itself is just its own vertex weight.
    dist[source] = A[source]
    heapq.heappush(pq, (dist[source], source))

    while pq:
        # Extract the vertex with the smallest distance found so far.
        current_cost, u = heapq.heappop(pq)

        # If we have found a shorter path to 'u' since this entry was added
        # to the queue, we can ignore this entry.
        if current_cost > dist[u]:
            continue

        # Iterate over all neighbors of the current vertex 'u'.
        for v, edge_weight in adj[u]:
            # The relaxation step is adapted for the problem's path weight definition.
            # Path weight to v through u = (path weight to u) + (edge u->v weight) + (vertex v weight).
            new_cost = dist[u] + edge_weight + A[v]

            # If this new path is shorter, update the distance and add it to the queue.
            if new_cost < dist[v]:
                dist[v] = new_cost
                heapq.heappush(pq, (new_cost, v))
                
    # The problem asks for results for vertices 2, 3, ..., N.
    # In 0-based indexing, this corresponds to indices 1, 2, ..., N-1.
    results = dist[1:]
    
    # Print the results as a single space-separated line.
    print(*results)

# Execute the solution
solve()