import sys

def main():
    N, P = map(int, sys.stdin.readline().split())
    k = N // 2

    # Precompute factorials and inverse factorials modulo P
    # Max value needed for combinations $\binom{n}{r}$ where n is up to k*k = (N/2)^2.
    # Also need combinations for choosing vertices: $\binom{k-1}{k-1}$ and $\binom{k}{k}$.
    # Max n for combinations seems to be k*k = (N/2)^2. For N=30, k=15, k*k=225.
    max_comb_n = k * k
    
    fact = [1] * (max_comb_n + 1)
    invfact = [1] * (max_comb_n + 1)
    for i in range(2, max_comb_n + 1):
        fact[i] = (fact[i - 1] * i) % P

    # Modular inverse for combinations
    invfact[max_comb_n] = pow(fact[max_comb_n], P - 2, P)
    for i in range(max_comb_n - 1, -1, -1):
        invfact[i] = (invfact[i + 1] * (i + 1)) % P

    def combinations(n, r):
        if r < 0 or r > n:
            return 0
        return (((fact[n] * invfact[r]) % P) * invfact[n - r]) % P

    # dp[i][j][m]: number of connected bipartite graphs on fixed partition sets A_0 (size i, includes vertex 1)
    # and B_0 (size j), with m edges.
    # The vertex sets are not specific {1..i}, {i+1..i+j}, but conceptually any fixed set of i vertices (containing 1)
    # and any fixed set of j vertices. The DP state is sufficient because the recurrence only depends on sizes.
    # dp[i][j][m] is 0 if i, j is not possible (e.g., i=0, j>0, connected and 1 in A0), or m outside [max(0, i+j-1), i*j].
    
    # dp[i][j][m] where i is size of part 1 (containing v1), j is size of part 2, m is edge count
    dp = [[[0 for _ in range(k * k + 1)] for _ in range(k + 1)] for _ in range(k + 1)]

    # Base case: A component with 1 vertex (vertex 1), 0 vertices in the other part, 0 edges.
    # This is a connected bipartite graph with partitions of size 1 and 0.
    # dp[1][0][0] means: connected bipartite graph with partition sizes 1 and 0, 0 edges, vertex 1 in the set of size 1.
    dp[1][0][0] = 1

    # Iterate over total number of vertices in the component s = i + j
    for s in range(1, N + 1):
        # Iterate over number of vertices in the first partition i (must contain vertex 1, so i >= 1)
        for i in range(1, min(s, k) + 1):
            j = s - i # Number of vertices in the second partition
            if j < 0 or j > k:
                continue

            # If s=1, we already handled the base case dp[1][0][0]=1. Continue to avoid recomputing or sum over empty ranges.
            if s == 1: continue

            # Iterate over number of edges m in the component
            # Minimum edges for connected component of size s > 1 is s - 1. Max edges is i*j.
            for m in range(max(0, s - 1), min(i * j, k * k) + 1):
                # Total bipartite graphs on fixed sets of size i, j with m edges is combinations(i*j, m)
                total_bipartite_m = combinations(i * j, m)

                # Subtract disconnected graphs where the component containing vertex 1 has size i' + j' < i + j
                disconnected_sum = 0
                
                # Iterate over possible sizes i', j' of the component containing vertex 1
                # i' vertices from the first partition (size i), j' from the second (size j)
                # Vertex 1 is in the component's first partition (size i'), so 1 <= i' <= i
                # 0 <= j' <= j
                for i_prime in range(1, i + 1):
                    for j_prime in range(0, j + 1):
                        # The component must be a proper subset of the current graph (size i+j)
                        if i_prime + j_prime < s:
                            # Number of ways to choose i' vertices (including 1) from the i vertices: combinations(i-1, i'-1)
                            # Number of ways to choose j' vertices from the j vertices: combinations(j, j')
                            choose_vertices_ways = (combinations(i - 1, i_prime - 1) * combinations(j, j_prime)) % P

                            # Iterate over possible number of edges m' in the component containing vertex 1
                            # m' must be sufficient for connectivity: >= max(0, i'+j'-1)
                            # m' cannot exceed possible edges within the component: <= i'*j'
                            # The remaining m-m' edges must fit in the remaining graph: m-m' <= (i-i')*(j-j')
                            
                            m_prime_lower_conn = max(0, i_prime + j_prime - 1) if i_prime + j_prime > 1 else 0
                            m_prime_lower_remaining = m - (i - i_prime) * (j - j_prime)

                            m_prime_lower = max(m_prime_lower_conn, m_prime_lower_remaining)
                            m_prime_upper = min(m, i_prime * j_prime)
                            
                            for m_prime in range(m_prime_lower, m_prime_upper + 1):
                                if dp[i_prime][j_prime][m_prime] > 0: # Optimization: only sum if connected component count is non-zero
                                     # Number of ways to choose m-m' edges in the remaining (i-i')*(j-j') slots
                                    remaining_edges_ways = combinations((i - i_prime) * (j - j_prime), m - m_prime)

                                    term = (choose_vertices_ways * dp[i_prime][j_prime][m_prime]) % P
                                    term = (term * remaining_edges_ways) % P
                                    disconnected_sum = (disconnected_sum + term) % P

                dp[i][j][m] = (total_bipartite_m - disconnected_sum + P) % P

    # Final Answer Calculation
    # The condition requires the graph to be connected and the distance partition from vertex 1
    # to have sizes N/2 and N/2. This implies the graph is bipartite with partition (V_even, V_odd)
    # where |V_even|=|V_odd|=N/2, and vertex 1 is in V_even.
    # We count graphs with fixed partition sizes k, k where vertex 1 is in the first partition.
    # Number of ways to choose the partition (A, B) with |A|=|B|=k and 1 in A.
    # A contains 1 and k-1 other vertices from N-1. Choose k-1 from N-1: combinations(N-1, k-1).
    # B contains the remaining k vertices.
    num_partitions = combinations(N - 1, k - 1)

    results = []
    max_edges_total_graph = N * (N - 1) // 2

    # For M from N-1 to N(N-1)/2
    for M in range(N - 1, max_edges_total_graph + 1):
        # A bipartite graph with partition sizes k, k can have at most k*k edges.
        # If the graph is required to be bipartite with partition sizes k, k,
        # then for M > k*k, the number of such graphs is 0.
        # The DP counts connected bipartite graphs on a FIXED k+k partition, with 1 in first part.
        # The condition implies such a structure.
        if M > k * k:
            ans = 0
        else:
            # The number of connected bipartite graphs with fixed partition sizes k, k and M edges,
            # with vertex 1 in the first partition is dp[k][k][M].
            # We multiplied by num_partitions to account for ways to choose WHICH sets of vertices form the partition.
            ans = (num_partitions * dp[k][k][M]) % P

        results.append(ans)

    print(*results)

if __name__ == "__main__":
    main()