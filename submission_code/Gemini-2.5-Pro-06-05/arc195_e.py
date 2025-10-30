import sys

def solve():
    """
    This function reads input, performs precomputation, and processes queries
    to solve the problem as described.
    """
    MOD = 998244353

    try:
        # Fast I/O
        line = sys.stdin.readline()
        if not line:
            return
        N, Q = map(int, line.split())
        A_list = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        # Handle potential empty input or parsing errors
        return

    # A_full is 1-indexed, so A_full[i] corresponds to A_i from the problem.
    # A_full[0] and A_full[1] are placeholders.
    A_full = [0, 0] + A_list

    # --- Precomputation Step ---

    # 1. Modular Inverses: O(N)
    # We need inverses for division in a modular field.
    inv = [0] * (N + 1)
    if N > 0:
        inv[1] = 1
    for i in range(2, N + 1):
        # Using the formula: inv[i] = MOD - (MOD // i) * inv[MOD % i] % MOD
        inv[i] = MOD - (MOD // i) * inv[MOD % i] % MOD

    # 2. Factorial: O(N)
    # The total number of trees is (N-1)!
    fact_N_minus_1 = 1
    for i in range(1, N):
        fact_N_minus_1 = (fact_N_minus_1 * i) % MOD

    # 3. Prefix Sums: O(N)
    # SA[i] = sum_{k=2 to i} A_k
    # SAB[i] = sum_{k=2 to i} A_k / k
    SA = [0] * (N + 1)
    SAB = [0] * (N + 1)
    for i in range(2, N + 1):
        SA[i] = (SA[i-1] + A_full[i]) % MOD
        term = (A_full[i] * inv[i]) % MOD
        SAB[i] = (SAB[i-1] + term) % MOD
    
    # --- Query Processing Step ---
    
    # Buffer output to print all at once for speed
    output = []
    for _ in range(Q):
        u, v = map(int, sys.stdin.readline().split())
        
        # Ensure u < v for consistency with the derived formulas
        if u > v:
            u, v = v, u

        # Let E[dist] be the expected distance over the uniform distribution of trees.
        # The total sum is (N-1)! * E[dist].
        # E[dist(u,v)] = E[dist(u,1)] + E[dist(v,1)] - 2*E[dist(LCA(u,v),1)]

        # Calculate E[dist(u, 1)]
        d_u_exp = 0
        if u > 1:
            d_u_exp = (A_full[u] + SAB[u-1]) % MOD

        # Calculate E[dist(v, 1)]
        d_v_exp = (A_full[v] + SAB[v-1]) % MOD

        # Calculate E[dist(LCA(u, v), 1)]. For u < v, this depends only on u.
        dl_u_exp = 0
        if u > 1:
            term1 = ((u + 1) * SAB[u-1]) % MOD
            term2 = SA[u-1]
            term3 = A_full[u]
            
            numerator = (term1 - term2 + term3 + MOD) % MOD
            dl_u_exp = (numerator * inv[u]) % MOD
            
        # Combine terms to get total expected distance
        total_exp_dist = (d_u_exp + d_v_exp - 2 * dl_u_exp) % MOD
        
        # Final answer is total trees * total expected distance
        ans = (fact_N_minus_1 * total_exp_dist) % MOD
        output.append(str(ans))

    sys.stdout.write('
'.join(output) + '
')

solve()