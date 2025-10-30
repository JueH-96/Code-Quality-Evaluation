# YOUR CODE HERE
import collections
import sys

# The main logic in a function
def solve():
    # Read number of vertices N and number of edges M
    N, M = map(int, sys.stdin.readline().split())
    
    # Build adjacency list using defaultdict for convenience
    # Stores tuples (neighbor_vertex, edge_weight)
    adj = collections.defaultdict(list)
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        # Use 0-indexed vertices internally for easier bitmask manipulation
        # Vertex i in input corresponds to index i-1
        adj[u - 1].append((v - 1, w))

    # Use a large integer value to represent infinity.
    # The maximum possible path sum could be large if many edges are traversed.
    # 10**18 is sufficiently large for typical competitive programming constraints.
    INF = 10**18 
    
    # Initialize DP table `dp[mask][u]`
    # `dp[mask][u]` stores the minimum total weight of a walk ending at vertex `u` (0-indexed),
    # having visited exactly the set of vertices represented by `mask`.
    # `mask` is an integer where the i-th bit is set if vertex i has been visited.
    # Dimensions: (1 << N) rows for masks, N columns for vertices.
    dp = [[INF] * N for _ in range(1 << N)]
    
    # Queue for the Shortest Path Faster Algorithm (SPFA).
    # Stores states as tuples `(mask, u)` representing the set of visited vertices and the last vertex.
    queue = collections.deque()
    
    # `in_queue[mask][u]` is a boolean flag indicating if state (mask, u) is currently in the queue.
    # This helps prevent adding duplicate states to the queue unnecessarily.
    # Dimensions are the same as the dp table.
    in_queue = [[False] * N for _ in range(1 << N)]

    # Initialize the base cases for SPFA.
    # For each vertex `i`, a walk starting and ending at `i` has visited only vertex `i`.
    # The mask representing this is `1 << i`, and the cost is 0.
    for i in range(N):
        mask = 1 << i
        dp[mask][i] = 0
        queue.append((mask, i))
        in_queue[mask][i] = True

    # SPFA main loop. Continues as long as there are states in the queue to process.
    while queue:
        # Dequeue the current state (mask, u)
        mask, u = queue.popleft()
        # Mark the state as removed from the queue
        in_queue[mask][u] = False 

        # Get the current minimum distance to this state
        current_dist = dp[mask][u]
        
        # Explore all edges outgoing from vertex u
        for v, w in adj[u]:
            # Calculate the mask for the new state after moving to vertex v
            # This involves setting the bit corresponding to vertex v
            new_mask = mask | (1 << v) 
            
            # Calculate the potential new distance to reach state (new_mask, v)
            new_dist = current_dist + w
            
            # Relaxation step: If the newly calculated distance is shorter than the existing distance
            if new_dist < dp[new_mask][v]:
                # Update the minimum distance
                dp[new_mask][v] = new_dist
                
                # If the state (new_mask, v) is not already in the queue, add it.
                # This ensures that we explore paths potentially improved by this shorter distance.
                if not in_queue[new_mask][v]:
                    queue.append((new_mask, v))
                    in_queue[new_mask][v] = True

    # After the SPFA algorithm completes, find the minimum weight walk that visits all vertices.
    # The mask representing all vertices visited is `(1 << N) - 1`.
    full_mask = (1 << N) - 1
    
    # Initialize minimum weight found so far to infinity.
    min_weight = INF
    
    # Iterate through all possible ending vertices `u` for walks that visited all vertices.
    for u in range(N):
        # Update `min_weight` with the minimum distance found.
        min_weight = min(min_weight, dp[full_mask][u])

    # Check if it was possible to visit all vertices.
    # If `min_weight` is still INF, it means no such walk exists.
    if min_weight == INF:
        print("No")
    else:
        # Otherwise, print the minimum weight found.
        # Since all input weights are integers, the minimum total weight will also be an integer.
        print(min_weight)

# Call the solver function to run the program
solve()