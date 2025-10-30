import itertools
import sys

def main():
    N = int(sys.stdin.readline())

    adj_G = [[False] * N for _ in range(N)]
    M_G = int(sys.stdin.readline())
    for _ in range(M_G):
        u, v = map(int, sys.stdin.readline().split())
        u -= 1 # Adjust to 0-indexing
        v -= 1 # Adjust to 0-indexing
        adj_G[u][v] = True
        adj_G[v][u] = True

    adj_H = [[False] * N for _ in range(N)]
    M_H = int(sys.stdin.readline())
    for _ in range(M_H):
        a, b = map(int, sys.stdin.readline().split())
        a -= 1 # Adjust to 0-indexing
        b -= 1 # Adjust to 0-indexing
        adj_H[a][b] = True
        adj_H[b][a] = True

    # costs_A[i][j] will store the cost for edge (i, j) with i < j (0-indexed)
    costs_A = [[0] * N for _ in range(N)] 
    # The input A_{i,j} are for 1-indexed i,j.
    # The first line has A_{1,2} A_{1,3} ... A_{1,N}
    # This corresponds to 0-indexed i=0, and j=1,2,...,N-1.
    # In general, for 0-indexed i_0idx, the input line gives costs for
    # (i_0idx, i_0idx+1), (i_0idx, i_0idx+2), ..., (i_0idx, N-1)
    for i_0idx in range(N - 1): # i_0idx from 0 to N-2
        row_input_costs = list(map(int, sys.stdin.readline().split()))
        for j_offset, cost_val in enumerate(row_input_costs):
            # The j_offset-th value in row_input_costs corresponds to
            # the pair (i_0idx, i_0idx + 1 + j_offset)
            j_0idx = i_0idx + 1 + j_offset
            costs_A[i_0idx][j_0idx] = cost_val
            
    min_total_cost = float('inf')
    
    vertex_indices = list(range(N)) # Original vertex indices [0, 1, ..., N-1]

    for P_tuple in itertools.permutations(vertex_indices):
        P = list(P_tuple) # P[g_idx] is the H-vertex that G-vertex g_idx maps to
        current_P_cost = 0
        
        # Iterate over all unique pairs of G's vertices (i, j) where i < j
        for i in range(N): 
            for j in range(i + 1, N): 
                # Edge status in G for pair (i,j)
                g_has_edge = adj_G[i][j] 
                
                # Corresponding vertices in H under permutation P
                h_vertex_mapped_from_i = P[i]
                h_vertex_mapped_from_j = P[j]
                
                # Get canonical representation for H edge (min_idx, max_idx)
                h_actual_node1_idx = min(h_vertex_mapped_from_i, h_vertex_mapped_from_j)
                h_actual_node2_idx = max(h_vertex_mapped_from_i, h_vertex_mapped_from_j)
                
                # Original edge status in H for this mapped pair
                h_has_edge = adj_H[h_actual_node1_idx][h_actual_node2_idx]
                
                # If G's requirement for edge (i,j) differs from H's current state for (P[i],P[j]),
                # we must pay to flip H's edge (h_actual_node1_idx, h_actual_node2_idx)
                if g_has_edge != h_has_edge:
                    current_P_cost += costs_A[h_actual_node1_idx][h_actual_node2_idx]
                    
        min_total_cost = min(min_total_cost, current_P_cost)
        
    # For N=1, loops for i,j won't run, current_P_cost remains 0. min_total_cost becomes 0. Correct.
    print(min_total_cost)

if __name__ == '__main__':
    main()