# YOUR CODE HERE
import sys
import itertools

# Use a sufficiently large integer value for infinity.
# The maximum possible path cost could be large. Given constraints:
# N <= 400, M <= 2e5, Q <= 3000, K <= 5, T <= 10^9.
# A path could involve K required bridges and K+1 connecting segments.
# Each segment could potentially traverse N-1 bridges.
# Max path sum could be roughly K * T_max + (K+1) * (N-1) * T_max.
# With K=5, N=400, T_max=10^9: 5*10^9 + 6*399*10^9 approx (5 + 2400) * 10^9 = 2405 * 10^9 ~ 2.4 * 10^12.
# A value like 10**18 is safely larger than any possible finite path sum.
INF = 10**18 

def solve():
    # Read number of islands N and bridges M
    N, M = map(int, sys.stdin.readline().split())

    # Store bridge information: (U, V, T) keyed by bridge index (1-based)
    bridge_info = {} 
    
    # Initialize distance matrix for Floyd-Warshall algorithm.
    # We use 1-based indexing for islands (nodes 1 to N).
    # Initialize distances to INF, except distance from a node to itself is 0.
    dist = [[INF] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        dist[i][i] = 0

    # Read bridge details and populate initial distance matrix.
    # Handle multiple bridges between the same pair of islands by taking the minimum time.
    for i in range(1, M + 1):
        u, v, t = map(int, sys.stdin.readline().split())
        bridge_info[i] = (u, v, t)
        dist[u][v] = min(dist[u][v], t)
        dist[v][u] = min(dist[v][u], t) # Bridges are bidirectional

    # Run Floyd-Warshall algorithm to compute all-pairs shortest paths.
    # This calculates the minimum time to travel between any two islands using any available bridges.
    for k in range(1, N + 1): # Intermediate node k
        for i in range(1, N + 1): # Start node i
            # Optimization: If i cannot reach k, path through k won't improve paths starting from i.
            if dist[i][k] == INF: continue 
            for j in range(1, N + 1): # End node j
                 # Optimization: If k cannot reach j, path through k won't improve paths ending at j.
                 if dist[k][j] == INF: continue 
                 
                 # Check if the path from i to j through k is shorter than the current shortest path.
                 new_dist = dist[i][k] + dist[k][j]
                 if new_dist < dist[i][j]:
                      dist[i][j] = new_dist

    # Read the number of queries
    Q = int(sys.stdin.readline())

    results = [] # List to store results for each query

    # Process each query
    for _ in range(Q):
        # Read query details: number of required bridges K and their indices B
        line = list(map(int, sys.stdin.readline().split()))
        K = line[0] 
        B = line[1:] # List of required bridge indices (1-based)

        min_total_time = INF # Initialize minimum time for this query to infinity

        # The problem requires finding a path from island 1 to island N that uses each bridge in B at least once.
        # We can model this as finding the minimum cost sequence of crossing the required bridges.
        # The path will look like: 1 -> ... -> s1 --B_p1--> d1 -> ... -> sK --B_pK--> dK -> ... -> N
        # where B_p1, ..., B_pK is a permutation of the required bridges B.
        # s_i, d_i are the start and end islands for crossing bridge B_pi.
        # The total cost is sum(T_pi) + sum of shortest path costs for segments between crossings.

        # Generate all permutations of the indices 0 to K-1.
        # These indices correspond to the positions in the required bridge list B.
        perm_indices = list(range(K))
        
        for p_indices in itertools.permutations(perm_indices):
            # p_indices represents an order to cross the required bridges. Example: if K=3, p_indices could be (1, 0, 2).
            
            # Iterate through all 2^K possible combinations of crossing directions for the K bridges.
            # Each bit in 'i' (from 0 to 2^K - 1) determines the direction for one bridge crossing.
            for i in range(1 << K): 
                
                total_bridge_times = 0 # Accumulates the times T_j for the required bridges
                path_segments_time = 0 # Accumulates the shortest path times between points
                
                crossings = [] # Stores tuples (start_node, end_node) for each required bridge crossing in the current order and direction choice
                
                # Determine the start and end nodes for each bridge crossing based on the permutation and direction choice.
                for k in range(K):
                    # Get the index into the B list for the k-th bridge in this permutation sequence
                    b_list_idx = p_indices[k] 
                    # Get the actual bridge index (1 to M) from the B list
                    bridge_actual_idx = B[b_list_idx] 
                    
                    U, V, T = bridge_info[bridge_actual_idx] # Get bridge details: endpoints U, V and time T
                    total_bridge_times += T # Add the time for crossing this bridge
                    
                    # Decide the direction of crossing based on the k-th bit of the direction mask 'i'.
                    # The k-th bit corresponds to the k-th bridge in the current permutation `p_indices`.
                    if (i >> k) & 1:
                        # If the k-th bit is 1, choose direction V -> U
                        s = V
                        d = U
                    else:
                        # If the k-th bit is 0, choose direction U -> V
                        s = U
                        d = V
                    crossings.append((s, d)) # Store the (start_node, end_node) pair for this crossing
                
                # Calculate the total time for path segments connecting the sequence of required bridge crossings.
                
                # Segment 1: From start island 1 to the start node of the first required bridge crossing.
                s_first = crossings[0][0]
                current_segment_time = dist[1][s_first] # Use precomputed shortest path distance
                
                # If this segment is impossible (distance is INF), this path configuration is invalid. Skip it.
                if current_segment_time == INF: 
                     continue 

                path_segments_time += current_segment_time # Add time for the first segment

                # Segments 2 to K: Between consecutive required bridge crossings.
                valid_intermediate_segments = True # Flag to track if all intermediate segments are possible
                for k in range(K - 1):
                    d_curr = crossings[k][1] # End node of the k-th crossing
                    s_next = crossings[k+1][0] # Start node of the (k+1)-th crossing
                    
                    current_segment_time = dist[d_curr][s_next] # Shortest path time between d_curr and s_next
                    
                    # If segment is impossible, mark the entire path invalid and break from this inner loop.
                    if current_segment_time == INF: 
                       valid_intermediate_segments = False
                       break 
                    
                    path_segments_time += current_segment_time # Add time for this intermediate segment

                # If any intermediate segment was impossible, skip to the next direction combination.
                if not valid_intermediate_segments: 
                   continue 
                
                # Segment K+1: From the end node of the last required bridge crossing to the destination island N.
                d_last = crossings[K - 1][1]
                current_segment_time = dist[d_last][N] # Shortest path time from d_last to N

                # If the final segment is impossible, this path configuration is invalid. Skip it.
                if current_segment_time == INF: 
                   continue 
                   
                path_segments_time += current_segment_time # Add time for the final segment

                # Calculate the total path time for this specific permutation and direction choice.
                current_path_time = total_bridge_times + path_segments_time
                
                # Update the minimum total time found so far for this query.
                min_total_time = min(min_total_time, current_path_time)

        # After checking all permutations and direction combinations, append the minimum time to the results list.
        # If no valid path was found (e.g., if some required bridge endpoint was disconnected from 1 or N, though problem guarantees connectivity), min_total_time would remain INF.
        results.append(min_total_time)

    # Print the results for all queries.
    for res in results:
        print(res)

# Execute the main function
solve()