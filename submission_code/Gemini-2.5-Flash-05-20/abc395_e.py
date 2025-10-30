import sys
import heapq

# This function encapsulates the solution logic
def solve():
    # Read N, M, X from the first line of input
    # N: number of vertices, M: number of edges, X: cost to reverse all edges
    N, M, X = map(int, sys.stdin.readline().split())

    # Initialize adjacency list for the state graph.
    # We will use 2*N nodes in our Dijkstra's algorithm:
    # Nodes 0 to N-1 represent vertices 1 to N in the original graph state.
    # Nodes N to 2*N-1 represent vertices 1 to N in the reversed graph state.
    #
    # Specifically, if an original vertex is `v` (1-indexed):
    # - Its corresponding node in the original state graph is `v-1`.
    # - Its corresponding node in the reversed state graph is `(v-1) + N`.
    adj = [[] for _ in range(2 * N)]

    # Populate adjacency list with edges for the 'move along a directed edge' operation (cost 1).
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        # Convert 1-indexed vertices to 0-indexed for array access
        u_idx = u - 1
        v_idx = v - 1

        # 1. Edge in the original graph state: u -> v with cost 1.
        # This translates to an edge from node `u_idx` to node `v_idx`.
        adj[u_idx].append((v_idx, 1))

        # 2. Edge in the reversed graph state:
        # An original edge u -> v means that in the reversed graph, there is an edge v -> u.
        # This translates to an edge from node `v_idx + N` to node `u_idx + N` with cost 1.
        adj[v_idx + N].append((u_idx + N, 1))

    # Populate adjacency list with edges for the 'reverse all directions' operation (cost X).
    # For each logical vertex `i` (0-indexed), we can switch between its original state
    # representation and its reversed state representation, and vice versa, both with cost X.
    for i in range(N):
        # From original state at vertex `i` to reversed state at vertex `i`
        # (Node `i` -> Node `i + N` with cost `X`)
        adj[i].append((i + N, X))
        
        # From reversed state at vertex `i` to original state at vertex `i`
        # (Node `i + N` -> Node `i` with cost `X`)
        adj[i + N].append((i, X))

    # Initialize distances for Dijkstra's algorithm.
    # `dist[k]` will store the minimum cost to reach node `k`.
    dist = [float('inf')] * (2 * N)

    # Priority queue for Dijkstra's, storing (cost, node_index) tuples.
    # `heapq` module implements a min-heap, so it will always pop the smallest cost path.
    pq = []

    # Set the starting point: vertex 1 (0-indexed as 0) in the original graph state.
    # The initial cost to reach this node is 0.
    dist[0] = 0
    heapq.heappush(pq, (0, 0)) # Push (initial_cost, start_node_index)

    # Main Dijkstra's algorithm loop
    while pq:
        current_cost, u = heapq.heappop(pq)

        # If we have already found a shorter path to node `u` than `current_cost`,
        # then this path is not optimal, so skip it.
        if current_cost > dist[u]:
            continue

        # Explore all neighbors `v` of the current node `u`.
        for v, edge_weight in adj[u]:
            # Calculate the total cost to reach neighbor `v` through the current path.
            new_cost = current_cost + edge_weight

            # If this new path to `v` is shorter than any previously found path to `v`
            if new_cost < dist[v]:
                dist[v] = new_cost       # Update the minimum cost to reach `v`
                heapq.heappush(pq, (new_cost, v)) # Push `v` to the priority queue

    # The goal is to reach vertex N (which is 0-indexed as N-1).
    # This can be achieved either in the original graph state (represented by node N-1)
    # or in the reversed graph state (represented by node N-1 + N).
    # We need the minimum of these two possibilities.
    result = min(dist[N - 1], dist[N - 1 + N])
    
    # Print the minimum total cost to standard output.
    # Using sys.stdout.write for potentially faster output.
    sys.stdout.write(str(result) + '
')

# Call the solve function to execute the program logic.
solve()