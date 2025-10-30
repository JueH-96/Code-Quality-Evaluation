# YOUR CODE HERE
import sys
import heapq
from collections import defaultdict

def solve():
    # Use faster input reading method
    readline = sys.stdin.readline

    # Read graph size N, number of edges M, and number of guards K
    N, M, K = map(int, readline().split())

    # Build adjacency list representation of the graph.
    # We use 0-based indexing internally for vertices (0 to N-1).
    adj = defaultdict(list)
    for _ in range(M):
        # Read edge endpoints u, v (1-based)
        u, v = map(int, readline().split())
        # Add edges to adjacency list using 0-based indices
        adj[u-1].append(v-1)
        adj[v-1].append(u-1)

    # Read guard information: location p (1-based) and stamina h
    initial_guards = []
    for _ in range(K):
        p, h = map(int, readline().split())
        # Store guard info with 0-based vertex index
        initial_guards.append(((p-1), h)) 

    # Initialize an array R to store the maximum remaining stamina found so far for each vertex.
    # Initialize all values to -1, indicating that vertices are initially not reached
    # or equivalently, reached with negative stamina (which means not guarded).
    R = [-1] * N
    
    # Use a max-priority queue to implement Dijkstra-like algorithm.
    # Since heapq is a min-heap, we store tuples (-stamina, vertex_idx).
    # The negative stamina ensures that extracting the minimum element effectively
    # gives us the element with the maximum stamina.
    pq = []

    # Initialize the priority queue and R array based on the initial guards.
    # For each guard i at vertex p_i with stamina h_i:
    # Set R[p_i] = h_i, as this is the maximum stamina available at the guard's starting location.
    # Add the guard's state (-h_i, p_i) to the priority queue.
    # Since the problem guarantees that all p_i are distinct, we don't need to worry about
    # multiple guards starting at the same vertex or taking the max of initial stamina values.
    for p, h in initial_guards:
        # Update the max stamina at vertex p
        R[p] = h
        # Push the initial state onto the priority queue
        heapq.heappush(pq, (-h, p))

    # Run the Dijkstra-like algorithm
    while pq:
        # Extract the state (vertex u) with the highest remaining stamina h
        neg_h, u = heapq.heappop(pq)
        h = -neg_h # Convert back to actual stamina value

        # Check for stale entries: If we have already found a path to u
        # with greater or equal remaining stamina, this entry is outdated.
        # We use h < R[u] check. If h == R[u], this is the current best path discovered
        # via the priority queue mechanism.
        if h < R[u]:
            continue

        # Optimization: If the current stamina is 0, we cannot propagate further
        # to neighbors with positive stamina. Any path extending from here would result
        # in negative remaining stamina at the neighbor. Since R is initialized to -1,
        # updating R[v] = -1 doesn't change anything useful.
        # We only care about vertices with R[v] >= 0 eventually.
        if h == 0:
             continue

        # Explore neighbors of the current vertex u
        for v in adj[u]:
            # Calculate the stamina if we move from u to neighbor v.
            # Each edge traversal costs 1 unit of stamina.
            new_stamina_at_v = h - 1
            
            # Relaxation step: Check if this path through u provides a higher
            # remaining stamina at vertex v than previously known.
            if new_stamina_at_v > R[v]:
                # Update the maximum stamina found for vertex v
                R[v] = new_stamina_at_v
                # Add the new state for v into the priority queue
                heapq.heappush(pq, (-new_stamina_at_v, v))

    # After the algorithm finishes, R[i] contains the maximum remaining stamina
    # with which vertex i can be reached from any guard.
    # A vertex i is guarded if R[i] >= 0.

    # Collect all guarded vertices
    guarded_vertices = []
    for i in range(N):
        if R[i] >= 0:
            # Append the 1-based index of the guarded vertex
            guarded_vertices.append(i + 1) 

    # Sort the guarded vertices in ascending order as required by the output specification
    guarded_vertices.sort()
    
    # Print the result in the specified format:
    # First line: number of guarded vertices (G)
    # Second line: space-separated list of guarded vertex indices (v_1 v_2 ... v_G)
    print(len(guarded_vertices))
    print(*(guarded_vertices)) # Use * operator to unpack the list into space-separated arguments for print

# Execute the main function to solve the problem
solve()

# END OF YOUR CODE HERE