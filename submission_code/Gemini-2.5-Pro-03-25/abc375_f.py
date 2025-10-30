# Import the sys module for faster input/output
import sys

# Set a large value to represent infinity. 
# Using a large integer is generally preferred over float('inf') in competitive programming
# to avoid potential precision issues and ensure compatibility with integer arithmetic.
# The maximum possible path length could be N * max(C_i) approx 300 * 10^9 = 3 * 10^11.
# 10**18 is sufficiently large.
INF_VAL = 10**18 

def solve():
    # Read input: Number of cities N, roads M, queries Q
    N, M, Q = map(int, sys.stdin.readline().split())

    # Store road information. We use a dictionary `roads_map` to map
    # the 1-based road index to its details (endpoints u, v and cost c).
    # City indices are converted to 0-based for internal use.
    roads_map = {}
    for i in range(M):
        # Read road details: city A_i, city B_i, cost C_i
        u, v, c = map(int, sys.stdin.readline().split())
        # Store road info with 0-based city indices
        road_info = {'id': i + 1, 'u': u - 1, 'v': v - 1, 'c': c}
        roads_map[i + 1] = road_info

    # Store all queries for later processing.
    queries = []
    for _ in range(Q):
        line = list(map(int, sys.stdin.readline().split()))
        queries.append(line)

    # Identify all roads that are eventually closed by type 1 queries.
    # Store their 1-based indices in a set for efficient lookup.
    closed_road_indices_set = set() 
    for i in range(Q):
        query = queries[i]
        # If query is type 1 (close road)
        if query[0] == 1:
            # Get the 1-based index of the road to be closed
            road_idx_1_based = query[1] 
            closed_road_indices_set.add(road_idx_1_based)

    # Initialize the distance matrix `dist`. This matrix will represent the state
    # of shortest paths *after* all road closures have occurred (chronologically).
    # Initially, all distances are infinity, except distance from a city to itself is 0.
    dist = [[INF_VAL] * N for _ in range(N)]
    for i in range(N):
        dist[i][i] = 0

    # Populate the initial distance matrix with edges corresponding to roads
    # that are *never* closed throughout the sequence of queries.
    for road_idx_1_based in range(1, M + 1):
        # If this road is not in the set of eventually closed roads
        if road_idx_1_based not in closed_road_indices_set:
            road = roads_map[road_idx_1_based]
            u, v, c = road['u'], road['v'], road['c']
            # Update distance if this road provides a shorter direct path.
            # Since roads are bidirectional, update both dist[u][v] and dist[v][u].
            dist[u][v] = min(dist[u][v], c)
            dist[v][u] = min(dist[v][u], c) 

    # Run the Floyd-Warshall algorithm on this initial graph state.
    # This computes all-pairs shortest paths considering only the roads that remain open
    # after all type 1 queries have been processed.
    for k in range(N): # Intermediate city
        for i in range(N): # Start city
            for j in range(N): # End city
                 # Check for infinity to prevent potential overflow if INF_VAL + INF_VAL occurs,
                 # though Python's arbitrary precision integers handle large numbers.
                 # This check primarily ensures we only consider valid paths.
                 if dist[i][k] != INF_VAL and dist[k][j] != INF_VAL:
                     # Update distance if path through k is shorter
                     dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    # Process queries in reverse order. This allows us to handle road closures
    # as road additions, which is easier to update shortest paths for.
    # `results` list will store the answers for type 2 queries.
    results = [None] * Q 

    # Iterate through queries from the last one (index Q-1) down to the first one (index 0).
    for q_idx in range(Q - 1, -1, -1):
        query = queries[q_idx]
        
        # If query is type 2 (find shortest path)
        if query[0] == 2: 
            # Get cities x and y, convert to 0-based indices
            x, y = query[1] - 1, query[2] - 1 
            # Get the current shortest distance from the `dist` matrix
            current_dist = dist[x][y]
            # If distance is infinity, it means y is unreachable from x. Store -1.
            if current_dist == INF_VAL:
                results[q_idx] = -1
            # Otherwise, store the computed distance.
            else:
                results[q_idx] = current_dist
        
        # If query is type 1 (close road)
        else: 
            # In reverse processing, a type 1 query means we "undo" the closure,
            # effectively adding the road back to the graph.
            road_idx_to_open = query[1] # 1-based index of the road to add back
            
            # Get the road details from the map.
            road = roads_map[road_idx_to_open]
            u, v, c = road['u'], road['v'], road['c']

            # Update the distance matrix using the O(N^2) edge addition algorithm.
            # We need to consider paths that might now become shorter by using the newly added edge (u, v).
            # Any such path must use the edge (u, v) exactly once (assuming non-negative costs).
            # A path from i to j using edge (u, v) looks like i -> ... -> u -> v -> ... -> j or i -> ... -> v -> u -> ... -> j.
            # The shortest such paths have lengths D[i][u] + c + D[v][j] and D[i][v] + c + D[u][j].
            
            # Create a snapshot `dist_old` of the current distances *before* this update step.
            # This is crucial to ensure correctness: all calculations for potential new paths
            # must use distances from the state *before* the edge was added.
            dist_old = [[dist[i][j] for j in range(N)] for i in range(N)]

            # Iterate over all pairs of cities (i, j) to update their shortest distance.
            for i in range(N):
                for j in range(N):
                     # Calculate potential path length via i -> u -> edge(u,v) -> v -> j
                     path1_len = INF_VAL
                     if dist_old[i][u] != INF_VAL and dist_old[v][j] != INF_VAL:
                         path1_len = dist_old[i][u] + c + dist_old[v][j]

                     # Calculate potential path length via i -> v -> edge(v,u) -> u -> j
                     path2_len = INF_VAL
                     if dist_old[i][v] != INF_VAL and dist_old[u][j] != INF_VAL:
                         path2_len = dist_old[i][v] + c + dist_old[u][j]

                     # Update dist[i][j] to the minimum of its old value and the lengths
                     # of the two potential new paths using the added edge.
                     dist[i][j] = min(dist_old[i][j], path1_len, path2_len)

    # After processing all queries in reverse, the `results` list contains
    # the answers for all type 2 queries at the time they were asked.
    # Print these results in the original query order.
    output_lines = []
    for i in range(Q):
        # Check if the i-th query was type 2
        if queries[i][0] == 2:
             # Append the stored result as a string
             output_lines.append(str(results[i]))

    # Write the final output to stdout, joining lines with newlines.
    sys.stdout.write("
".join(output_lines) + "
")

# Execute the solve function
solve()