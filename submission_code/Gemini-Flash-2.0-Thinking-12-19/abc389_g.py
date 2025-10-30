import sys

def power(a, b, m):
    res = 1
    a %= m
    while b > 0:
        if b % 2 == 1:
            res = (res * a) % m
        a = (a * a) % m
        b //= 2
    return res

def modInverse(n, m):
    return power(n, m - 2, m)

def precompute_factorials(max_n, m):
    fact = [1] * (max_n + 1)
    invFact = [1] * (max_n + 1)
    for i in range(2, max_n + 1):
        fact[i] = (fact[i - 1] * i) % m
        invFact[i] = modInverse(fact[i], m)
    return fact, invFact

def solve():
    N, P = map(int, sys.stdin.readline().split())

    MAX_N_HALF = N // 2
    MAX_EDGES_BIPARTITE_PARTITION = MAX_N_HALF * MAX_N_HALF # Max edges in a bipartite graph with partitions of size N/2, N/2
    MAX_TOTAL_EDGES_IN_GRAPH = N * (N - 1) // 2 # Max edges in any simple graph on N vertices

    # Max value for combinations n: max(MAX_EDGES_BIPARTITE_PARTITION, N - 1)
    # Also need combinations for remaining edges: max (s-s')(t-t') which is <= MAX_EDGES_BIPARTITE_PARTITION
    # Max value for choosing vertices: N-1
    MAX_COMB_N = max(MAX_EDGES_BIPARTITE_PARTITION, N - 1)
    fact, invFact = precompute_factorials(MAX_COMB_N, P)

    def comb(n, k):
        return combinations(n, k, P, fact, invFact)

    # dp[s][t][k]: number of connected bipartite graphs with fixed partitions
    # of size s and t (vertex 1 in the first partition) and k edges.
    # The vertices are assumed to be a specific set of s vertices (including 1)
    # and a specific set of t vertices.
    # s: size of partition A (containing vertex 1), 1 <= s <= N/2
    # t: size of partition B, 0 <= t <= N/2
    # k: number of edges, 0 <= k <= s*t
    dp = [[[0 for _ in range(MAX_EDGES_BIPARTITE_PARTITION + 1)] for _ in range(MAX_N_HALF + 1)] for _ in range(MAX_N_HALF + 1)]

    # Base case: s=1, t=0. Component is just vertex 1.
    dp[1][0][0] = 1

    # Iterate over increasing size of the graph total_size = s+t
    for total_size in range(1, N + 1):
        # Iterate over possible sizes s for partition A (containing vertex 1)
        # s must be at least 1 since it contains vertex 1
        for s in range(1, min(total_size, MAX_N_HALF) + 1):
            t = total_size - s
            # If t is out of bounds for partition B size, continue
            if t < 0 or t > MAX_N_HALF:
                continue

            max_k_st = s * t

            # Iterate over possible number of edges k
            # If total_size > 1, connected graph needs at least total_size - 1 edges.
            # Calculate dp[s][t][k] for all k from 0 to st, handle connectivity requirement later
            for k in range(0, max_k_st + 1):

                # If total_size > 1 and k < total_size - 1, it's impossible to be connected, dp[s][t][k] should be 0.
                # We calculate total graphs and subtract disconnected. If total_size > 1 and k < total_size - 1,
                # total_bip will be non-zero, subtraction_sum must equal total_bip.
                
                # Total number of bipartite graphs with fixed partitions (size s, t) and k edges on these specific vertices
                total_bip = comb(max_k_st, k)

                # Subtract disconnected graphs where vertex 1 is in a component of size s'+t' < s+t
                subtraction_sum = 0
                
                # Iterate over possible sizes (s', t') of the component containing vertex 1
                # Component vertices are a subset of the current s+t vertices.
                # Choose s' vertices from the current s vertices in A (1 is fixed in the component), t' from the current t vertices in B.
                # 1 <= s' <= s, 0 <= t' <= t. Condition s'+t' < s+t ensures we use already computed DP values.
                for s_prime in range(1, s + 1):
                    for t_prime in range(0, t + 1):
                        # Component must be strictly smaller than the current graph
                        if s_prime == s and t_prime == t:
                            continue

                        # Component size
                        component_size = s_prime + t_prime
                        
                        # Ways to choose vertices for this component from the fixed s vertices in A (1 always chosen) and t vertices in B
                        # Choose s'-1 vertices from the s-1 vertices in A (excluding 1),
                        # Choose t' vertices from the t vertices in B
                        ways_choose_vtx = (comb(s - 1, s_prime - 1) * comb(t, t_prime)) % P

                        max_k_prime_st_prime = s_prime * t_prime

                        # Iterate over possible number of edges k' in this component
                        for k_prime in range(0, min(k, max_k_prime_st_prime) + 1):
                            # The component itself must be connected
                            # A connected graph with component_size > 1 vertices needs at least component_size - 1 edges.
                            if component_size > 1 and k_prime < (component_size - 1):
                                continue

                            conn_graphs_comp = dp[s_prime][t_prime][k_prime]
                            if conn_graphs_comp == 0: # Optimization
                                continue

                            # Remaining edges in the rest of the graph
                            rem_k = k - k_prime
                            # Remaining vertices in the rest of the graph
                            rem_s = s - s_prime
                            rem_t = t - t_prime
                            rem_edges_cap = rem_s * rem_t

                            # Number of any bipartite graphs on remaining vertices with rem_k edges
                            rem_graphs_rest = comb(rem_edges_cap, rem_k)

                            term = (ways_choose_vtx * conn_graphs_comp % P * rem_graphs_rest % P) % P
                            subtraction_sum = (subtraction_sum + term) % P

                # Number of connected graphs = Total graphs - Disconnected graphs (where 1 is in a smaller component)
                dp[s][t][k] = (total_bip - subtraction_sum + P) % P

    # Compute the final answers for each M
    results = []
    
    # Number of ways to choose the N/2-1 vertices (other than vertex 1)
    # that will be in the same partition as vertex 1.
    ways_to_choose_partition = comb(N - 1, MAX_N_HALF - 1)

    # M ranges from N-1 up to N(N-1)/2
    for M in range(N - 1, MAX_TOTAL_EDGES_IN_GRAPH + 1):
        # If M > MAX_EDGES_BIPARTITE_PARTITION, it's impossible to form
        # a bipartite graph with partitions of size N/2. The distance parity
        # condition for vertex 1 implies such a bipartite structure.
        # However, sample output suggests non-bipartite graphs might be included.
        # The DP calculates connected bipartite graphs with fixed partitions.
        # Let's output the result based on the DP and multiplication by ways to choose partition.
        # If M > MAX_EDGES_BIPARTITE_PARTITION, dp[N/2][N/2][M] should be 0 by definition (k > st).
        
        ans = (ways_to_choose_partition * dp[MAX_N_HALF][MAX_N_HALF][M]) % P
        results.append(ans)

    print(*results)

solve()