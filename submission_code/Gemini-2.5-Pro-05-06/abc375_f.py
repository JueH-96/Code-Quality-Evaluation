import sys

def solve():
    N, M, Q = map(int, sys.stdin.readline().split())

    all_roads = []
    # Adjust to 0-indexed when reading roads
    for _ in range(M):
        u, v, c = map(int, sys.stdin.readline().split())
        all_roads.append((u - 1, v - 1, c)) 

    # Keep track of which roads (by their original index) are closed
    is_road_closed = [False] * M
    
    dist_apsp = None # This will store the all-pairs shortest paths matrix
    needs_recalc = True # Flag to indicate if dist_apsp needs to be recomputed

    output_buffer = [] # Store results to print all at once

    for _ in range(Q):
        query_line = sys.stdin.readline().split() 
        query_type = int(query_line[0])

        if query_type == 1:
            road_idx_1_based = int(query_line[1])
            # Mark the road as closed (using 0-indexed road index)
            # Problem guarantees road is not already closed, so no check needed here.
            is_road_closed[road_idx_1_based - 1] = True
            needs_recalc = True # Graph has changed, so paths may need recomputation
        else: # query_type == 2
            u_1_based, v_1_based = int(query_line[1]), int(query_line[2])
            # Convert to 0-indexed for matrix access
            u, v = u_1_based - 1, v_1_based - 1 

            if needs_recalc:
                # Initialize distance matrix for Floyd-Warshall
                current_dist_matrix = [[float('inf')] * N for _ in range(N)]
                for i in range(N):
                    current_dist_matrix[i][i] = 0 # Distance to self is 0
                
                # Add edges that are not closed
                for r_idx in range(M):
                    if not is_road_closed[r_idx]:
                        r_u, r_v, r_c = all_roads[r_idx]
                        current_dist_matrix[r_u][r_v] = min(current_dist_matrix[r_u][r_v], r_c)
                        current_dist_matrix[r_v][r_u] = min(current_dist_matrix[r_v][r_u], r_c)
                
                # Floyd-Warshall algorithm
                for k_fw in range(N): # Intermediate vertex
                    for i_fw in range(N): # Source vertex
                        for j_fw in range(N): # Destination vertex
                            # If path through k_fw is shorter
                            # Check for float('inf') to avoid issues with inf+inf arithmetic if any value could be negative (not here)
                            # and to ensure valid paths are being combined.
                            if current_dist_matrix[i_fw][k_fw] != float('inf') and \
                               current_dist_matrix[k_fw][j_fw] != float('inf'):
                                new_dist_val = current_dist_matrix[i_fw][k_fw] + current_dist_matrix[k_fw][j_fw]
                                if new_dist_val < current_dist_matrix[i_fw][j_fw]:
                                    current_dist_matrix[i_fw][j_fw] = new_dist_val
                
                dist_apsp = current_dist_matrix # Store the computed APSP matrix
                needs_recalc = False # Mark as up-to-date

            # Retrieve the shortest distance from the (possibly cached) APSP matrix
            ans_dist = dist_apsp[u][v]
            
            if ans_dist == float('inf'):
                output_buffer.append("-1")
            else:
                output_buffer.append(str(int(ans_dist))) # Distances are integers

    # Print all results, each on a new line
    sys.stdout.write("
".join(output_buffer))
    if output_buffer: # Ensure a final newline if there was any output
         sys.stdout.write("
")

solve()