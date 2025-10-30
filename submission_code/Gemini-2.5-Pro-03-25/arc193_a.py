# YOUR CODE HERE
import sys
import heapq

# By default recursion depth is 1000. Dijkstra is iterative, but deep graph traversals
# might trigger system limits in other ways. Set higher just in case if needed, 
# but probably not necessary for iterative Dijkstra.
# sys.setrecursionlimit(200000) 

def solve():
    # Read input N
    N = int(sys.stdin.readline())
    
    # Read weights W_1 ... W_N
    # Store as 0-indexed list W, where W[i] is the weight of vertex i+1
    W = list(map(int, sys.stdin.readline().split()))
    
    # Read N intervals [L_i, R_i]
    # Store intervals along with their original 0-based index
    intervals = []
    for i in range(N):
        # Read L_i, R_i. Store as [L_i, R_i, original_index]
        intervals.append(list(map(int, sys.stdin.readline().split())) + [i]) 

    # Read number of queries Q
    Q = int(sys.stdin.readline())
    
    # Read Q queries (s_i, t_i)
    # Store queries along with their original query index to output in correct order
    queries = []
    for i in range(Q):
        # Read s_i, t_i. Store as [s_i, t_i, query_index]
        queries.append(list(map(int, sys.stdin.readline().split())) + [i]) 

    # Build the augmented graph
    # The coordinate range is [1, 2N]. We use auxiliary vertices corresponding to indices 0 to 2N+1.
    # Total number of auxiliary coordinates points needed is 2N+2 (indices 0, 1, ..., 2N+1).
    num_aux_coords = 2 * N + 2 
    
    # Total number of vertices in the augmented graph:
    # N interval vertices (indices 0 to N-1)
    # 2 * num_aux_coords auxiliary vertices (one set for x chain, one for y chain)
    num_vertices = N + 2 * num_aux_coords 
    
    # Initialize adjacency list for the augmented graph
    adj = [[] for _ in range(num_vertices)]

    # Define base indices for different types of nodes in the adjacency list
    # Interval nodes use indices 0 to N-1 (their original 0-based index)
    x_base_idx = N # x_k vertices use indices N to N + num_aux_coords - 1
    y_base_idx = N + num_aux_coords # y_k vertices use indices N + num_aux_coords to N + 2*num_aux_coords - 1

    # Add edges for the x chain: x_k -> x_{k+1} with weight 0, for k = 0 to 2N
    for k in range(num_aux_coords - 1):
        u = x_base_idx + k
        v = x_base_idx + k + 1
        adj[u].append((v, 0)) # Edge (u, v) with weight 0

    # Add edges for the y chain: y_{k+1} -> y_k with weight 0, for k = 0 to 2N
    for k in range(num_aux_coords - 1):
        u = y_base_idx + k + 1
        v = y_base_idx + k
        adj[u].append((v, 0)) # Edge (u, v) with weight 0

    # Add edges connecting interval vertices to auxiliary coordinate vertices
    for i in range(N):
        L_i, R_i, original_idx = intervals[i] # L_i, R_i are 1-based coordinates
        W_i = W[original_idx] # Get weight using original index
        
        interval_node = original_idx # The node index for this interval vertex

        # Edge from x_{L_i-1} to interval_node with weight W_i
        # Represents paths ending at interval i from the left
        coord_idx_x_in = L_i - 1 # Convert L_i to 0-based coordinate index
        if 0 <= coord_idx_x_in < num_aux_coords: # Check if coordinate index is valid [0, 2N+1]
             x_u = x_base_idx + coord_idx_x_in # Calculate vertex index for x_{L_i-1}
             adj[x_u].append((interval_node, W_i)) # Add edge
        
        # Edge from interval_node to x_{R_i} with weight 0
        # Represents leaving interval i towards the right
        coord_idx_x_out = R_i # R_i is 1-based coordinate index, directly use as coordinate point index
        if 0 <= coord_idx_x_out < num_aux_coords:
            x_v = x_base_idx + coord_idx_x_out # Calculate vertex index for x_{R_i}
            adj[interval_node].append((x_v, 0)) # Add edge

        # Edge from y_{R_i+1} to interval_node with weight W_i
        # Represents paths ending at interval i from the right
        coord_idx_y_in = R_i + 1 # Coordinate index for point right after R_i
        if 0 <= coord_idx_y_in < num_aux_coords: 
            y_u = y_base_idx + coord_idx_y_in # Calculate vertex index for y_{R_i+1}
            adj[y_u].append((interval_node, W_i)) # Add edge

        # Edge from interval_node to y_{L_i} with weight 0
        # Represents leaving interval i towards the left
        coord_idx_y_out = L_i # L_i is 1-based coordinate index, directly use as coordinate point index
        if 0 <= coord_idx_y_out < num_aux_coords:
            y_v = y_base_idx + coord_idx_y_out # Calculate vertex index for y_{L_i}
            adj[interval_node].append((y_v, 0)) # Add edge

    # Prepare list to store results for each query
    results = [""] * Q 

    # Process queries one by one using Dijkstra's algorithm
    for s_orig, t_orig, query_idx in queries:
        # Convert 1-based query indices to 0-based indices used in the graph
        s = s_orig - 1 
        t = t_orig - 1 

        # Initialize distances list for Dijkstra. Use float('inf') for unreachable nodes.
        dist = [float('inf')] * num_vertices
        
        # Initialize priority queue for Dijkstra.
        # The path weight definition includes the starting vertex weight W_s.
        # Start Dijkstra with initial distance W[s] for the source node s.
        pq = [(W[s], s)] 
        dist[s] = W[s]

        # Store the final shortest distance found to target t
        final_dist_to_t = float('inf') 

        while pq:
            # Extract vertex u with the smallest distance d from priority queue
            d, u = heapq.heappop(pq)

            # If we extracted a path with distance d, but we already found a shorter path to u, skip.
            if d > dist[u]:
                continue
            
            # Optimization: If the extracted node `u` is the target node `t`,
            # its distance `d` is the minimum possible path weight. We can stop Dijkstra.
            if u == t:
                 final_dist_to_t = d
                 break 

            # Relax edges outgoing from u
            for v, weight in adj[u]:
                 # Calculate distance through u to neighbor v
                 new_dist = d + weight
                 
                 # If this path is shorter than the known shortest path to v
                 if new_dist < dist[v]:
                     # Update distance and push/update v in priority queue
                     dist[v] = new_dist
                     heapq.heappush(pq, (new_dist, v))
        
        # After Dijkstra finishes for this query (either PQ empty or target t finalized)
        # Check the final distance computed for t
        if final_dist_to_t == float('inf'):
             # If t was never reached or finalized with finite distance, no path exists.
             results[query_idx] = "-1"
        else:
             # Otherwise, the minimum path weight is final_dist_to_t
             results[query_idx] = str(final_dist_to_t)
             
    # Print results for all queries in their original order
    print('
'.join(results))

# Execute the solve function
solve()