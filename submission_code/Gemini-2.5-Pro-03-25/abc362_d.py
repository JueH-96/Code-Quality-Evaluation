# YOUR CODE HERE
import sys
import heapq

# Function to solve the problem
def solve():
    # Read N (number of vertices) and M (number of edges) from input
    N, M = map(int, sys.stdin.readline().split())
    
    # Read vertex weights A_i. Store them in a list A.
    # We use 1-based indexing for vertices to match the problem statement.
    # A[0] is unused padding, A[1]...A[N] store the weights.
    A = [0] + list(map(int, sys.stdin.readline().split())) 

    # Initialize adjacency list for the graph.
    # `adj[i]` will store a list of tuples `(neighbor, edge_weight)` for vertex `i`.
    # Use 1-based indexing for vertices, so the list size is N+1.
    adj = [[] for _ in range(N + 1)] 
    
    # Read M edges and build the adjacency list.
    # The graph is undirected, so for an edge between u and v with weight b,
    # we add v to u's neighbors and u to v's neighbors.
    for _ in range(M):
        u, v, b = map(int, sys.stdin.readline().split())
        adj[u].append((v, b))
        adj[v].append((u, b))

    # Initialize distance array `dist`. `dist[i]` will store the minimum path weight found so far
    # from vertex 1 to vertex i. Use 1-based indexing (size N+1).
    # Initialize all distances to infinity. Python's float('inf') is suitable.
    dist = [float('inf')] * (N + 1)
    
    # The minimum path weight from the source vertex 1 to itself is defined by the path containing only vertex 1.
    # The weight of this path is simply the weight of vertex 1, A[1].
    dist[1] = A[1]

    # Initialize a min-priority queue using Python's heapq module.
    # Store tuples (distance, vertex_index). The priority queue will order elements based on distance.
    # Start with the source vertex 1 and its initial distance A[1].
    pq = [(A[1], 1)]

    # Dijkstra's algorithm main loop. Continues as long as there are vertices to process in the priority queue.
    while pq:
        # Extract the vertex `u` with the smallest current distance `d` from the priority queue.
        d, u = heapq.heappop(pq)

        # Optimization: If the extracted distance `d` is greater than the already known shortest distance
        # to `u` stored in `dist[u]`, this means we have found a shorter path to `u` already and processed it.
        # This current entry `(d, u)` is outdated, so we skip it and proceed to the next element in the queue.
        if d > dist[u]:
            continue

        # Iterate through all neighbors `v` of the current vertex `u`.
        # `b` is the weight of the edge (u, v).
        for v, b in adj[u]:
            # Calculate the potential new distance to neighbor `v` by extending the path to `u`.
            # According to the problem definition, the path weight increases by the edge weight `b`
            # and the weight of the newly added vertex `v`, which is `A[v]`.
            # The total weight of the path $1 \leadsto u \to v$ is $dist[u] + b + A[v]$.
            new_dist = dist[u] + b + A[v]
            
            # Relaxation step: If this newly calculated path to `v` through `u` is shorter
            # than the currently known shortest path distance to `v` (`dist[v]`),
            # update `dist[v]` with this `new_dist`.
            if new_dist < dist[v]:
                dist[v] = new_dist
                # Add the updated information (new shorter distance, vertex v) to the priority queue.
                # This ensures that neighbors of `v` will be explored later starting from this improved distance.
                # If `v` was already in the queue with a larger distance, this new entry will be processed first.
                heapq.heappush(pq, (new_dist, v))

    # After Dijkstra's algorithm completes, the `dist` array contains the minimum path weights
    # from vertex 1 to all other vertices in the graph.
    # The problem asks for the minimum path weights for vertices 2 through N.
    
    # Collect the required distances into a list of strings.
    result = []
    for i in range(2, N + 1):
        # Python integers handle arbitrary size, so large weights are okay.
        # Convert the integer distance `dist[i]` to a string for printing.
        result.append(str(dist[i])) 

    # Print the results as a single line of space-separated values to standard output.
    print(" ".join(result))

# Call the solve function to execute the program logic.
solve()