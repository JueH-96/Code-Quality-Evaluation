import sys

def solve():
    N, P = map(int, sys.stdin.readline().split())

    # Maximum possible edges in a graph with N vertices
    MAX_E = N * (N - 1) // 2
    
    # Precompute factorials and inverse factorials modulo P
    # Required for nCr(n, r) where r can be up to MAX_E
    fact = [1] * (MAX_E + 1)
    inv_fact = [1] * (MAX_E + 1)
    for i in range(1, MAX_E + 1):
        fact[i] = (fact[i-1] * i) % P
    
    # inv_fact[MAX_E] = fact[MAX_E]^(P-2) mod P
    inv_fact[MAX_E] = pow(fact[MAX_E], P - 2, P)
    # inv_fact[i] = inv_fact[i+1] * (i+1) mod P
    for i in range(MAX_E - 1, -1, -1):
        inv_fact[i] = (inv_fact[i+1] * (i+1)) % P

    def nCr_mod_p(n, r):
        # Handles cases where r < 0 or r > n
        if r < 0 or r > n:
            return 0
        # nCr(X, 0) = 1
        if r == 0 or r == n:
            return 1
        
        # Optimize for r > n/2 to calculate nC(n-r)
        if r > n // 2:
            r = n - r
        
        # Calculate n*(n-1)*...*(n-r+1) mod P
        numerator = 1
        for i in range(r):
            numerator = (numerator * (n - i)) % P
        
        # Multiply by (r!)^(-1) mod P
        denominator_inv = inv_fact[r] 
        
        return (numerator * denominator_inv) % P

    # dp[num_total_vertices][num_even_dist_vertices][num_edges]
    # Stores the number of connected graphs where vertex 1 is the root.
    # num_total_vertices (i): 1 to N
    # num_even_dist_vertices (j): 1 to i (as vertex 1 is always even)
    # num_edges (k): 0 to MAX_E
    dp = [[[0] * (MAX_E + 1) for _ in range(N + 1)] for _ in range(N + 1)]

    # Base case: Graph with only vertex 1.
    # It has 1 vertex, 1 even-distance vertex (itself), and 0 edges.
    dp[1][1][0] = 1 

    # Iterate over total number of vertices 'i' in the component of vertex 1
    for i in range(1, N + 1):
        # Iterate over number of even-distance vertices 'j' in this component
        # 'j' must be at least 1 because vertex 1 itself is in V_e.
        for j in range(1, i + 1):
            num_odd_chosen = i - j # Number of odd-distance vertices in this component
            
            # Calculate total possible edges that can exist given this (j, num_odd_chosen) partition.
            # Edges can be within V_e, within V_o, or between V_e and V_o.
            total_possible_edges_for_partition = \
                j * (j - 1) // 2 + \
                num_odd_chosen * (num_odd_chosen - 1) // 2 + \
                j * num_odd_chosen
            
            # Iterate over number of edges 'k' in this component
            for k in range(0, min(MAX_E + 1, total_possible_edges_for_partition + 1)):
                # Skip the base case, already initialized
                if i == 1 and j == 1 and k == 0:
                    continue

                # current_total_graphs: total number of graphs (not necessarily connected)
                # on 'i' vertices with 'k' edges, respecting the (j, num_odd_chosen) partition.
                current_total_graphs = nCr_mod_p(total_possible_edges_for_partition, k)

                subtraction_sum = 0
                # Iterate over possible sizes of the component containing vertex 1, 'c_i'.
                # 'c_i' must be less than 'i' for a disconnected component.
                for c_i in range(1, i):
                    # Iterate over count of even-dist vertices in 1's component, 'c_j'.
                    # 'c_j' must be at least 1 (for vertex 1) and not exceed 'j'.
                    for c_j in range(1, j + 1):
                        # Calculate remaining vertices counts in the (j, num_odd_chosen) partition
                        rem_total_vertices = i - c_i
                        rem_even_vertices = j - c_j
                        rem_odd_vertices = num_odd_chosen - (c_i - c_j)

                        # If vertex counts are invalid, skip
                        if rem_total_vertices < 0 or rem_even_vertices < 0 or rem_odd_vertices < 0:
                            continue

                        # num_ways_to_choose_vertices: Number of ways to choose vertices for 1's component
                        # from the 'i' vertices (excluding 1).
                        # We choose (c_j - 1) 'even' vertices from (j - 1) available 'even' vertices
                        # and (c_i - c_j) 'odd' vertices from (num_odd_chosen) available 'odd' vertices.
                        num_ways_to_choose_vertices = (nCr_mod_p(j - 1, c_j - 1) * nCr_mod_p(num_odd_chosen, c_i - c_j)) % P
                        
                        # Calculate total possible edges for the remaining vertices' partition.
                        # This is used for `graphs_on_remaining`.
                        tpe_rem = \
                            rem_even_vertices * (rem_even_vertices - 1) // 2 + \
                            rem_odd_vertices * (rem_odd_vertices - 1) // 2 + \
                            rem_even_vertices * rem_odd_vertices

                        # Sum over possible number of edges 'c_k' in 1's component.
                        for c_k in range(0, k + 1):
                            # Optimization: if no such connected graph, skip
                            if dp[c_i][c_j][c_k] == 0:
                                continue 

                            rem_k = k - c_k # Edges remaining for the other part of the graph

                            # graphs_on_remaining: Number of ways to pick edges for the remaining vertices' partition
                            # This part is treated as an arbitrary graph on the remaining subsets of vertices.
                            graphs_on_remaining = nCr_mod_p(tpe_rem, rem_k)
                            
                            # Add to subtraction_sum: (ways to choose vertices) * (connected graphs in 1's component) * (graphs in remaining part)
                            term = (dp[c_i][c_j][c_k] * graphs_on_remaining) % P
                            subtraction_sum = (subtraction_sum + num_ways_to_choose_vertices * term) % P
                
                # Calculate dp[i][j][k] using the inclusion-exclusion principle
                dp[i][j][k] = (current_total_graphs - subtraction_sum + P) % P
    
    results_list = []
    # Loop for M values from N-1 (min edges for connected) to MAX_E (max edges)
    for m_val in range(N - 1, MAX_E + 1):
        # As per problem: N is even. We require N/2 vertices with even dist, N/2 with odd dist.
        # This means `j` must be exactly N/2.
        required_even_dist_count = N // 2
        
        # Append the calculated count for current M to the results list
        results_list.append(dp[N][required_even_dist_count][m_val])

    # Output the results, space-separated, on a single line
    sys.stdout.write(" ".join(map(str, results_list)) + "
")

solve()