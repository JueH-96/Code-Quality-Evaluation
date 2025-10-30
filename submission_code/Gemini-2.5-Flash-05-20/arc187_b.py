import sys

# Define the modulo constant
MOD = 998244353

def solve():
    N, M = map(int, sys.stdin.readline().split())
    B = list(map(int, sys.stdin.readline().split()))

    # Precompute num_neg1_prefix and min_fixed_prefix arrays.
    # num_neg1_prefix[k]: count of -1s in B[0...k-1]
    # min_fixed_prefix[k]: minimum value among B[j] for j in [0...k-1] where B[j] != -1.
    # It's initialized to M + 1, a value greater than any possible M.
    num_neg1_prefix = [0] * (N + 1)
    min_fixed_prefix = [M + 1] * (N + 1)

    for k in range(N):
        num_neg1_prefix[k+1] = num_neg1_prefix[k]
        min_fixed_prefix[k+1] = min_fixed_prefix[k]
        if B[k] == -1:
            num_neg1_prefix[k+1] += 1
        else:
            min_fixed_prefix[k+1] = min(min_fixed_prefix[k+1], B[k])

    # Precompute powers[val][exp] = val^exp % MOD.
    # `val` ranges from 0 to M. `exp` ranges from 0 to N-1.
    # Standard combinatorial convention: 0^0 = 1. For exp > 0, 0^exp = 0.
    pows = [[0] * N for _ in range(M + 1)]
    for v in range(M + 1):
        pows[v][0] = 1 # v^0 = 1 for all v (including v=0)
        if v != 0:
            for e in range(1, N):
                pows[v][e] = (pows[v][e-1] * v) % MOD
        # If v == 0, pows[0][0] is 1, and pows[0][e] for e > 0 remain 0 (as initialized), which is correct.

    # total_q: total number of -1s in the entire sequence B.
    total_q = num_neg1_prefix[N]

    # Initialize the total sum.
    # Vertex 1 is always a component root (no j < 1).
    # Its contribution is the total number of ways to fill B', which is M^total_q.
    ans = pows[M][total_q] 

    # Iterate for i from 2 to N (0-indexed B[i-1]).
    # We calculate the count of B' sequences where vertex `i` is a root.
    for i in range(2, N + 1):
        # q_prev: number of -1s in B[0...i-2] (elements before B[i-1]).
        q_prev = num_neg1_prefix[i-1] 
        # m_fixed: minimum fixed value in B[0...i-2].
        m_fixed = min_fixed_prefix[i-1] 

        term_i = 0 # This will store the number of ways to fill B[0...i-1]
                   # such that B[j] > B[i-1] for all j < i-1.
        
        if B[i-1] != -1: # Case 1: B[i-1] is a fixed integer value `x`.
            x = B[i-1]
            # For vertex `i` to be a root, all A_j (for j < i) must be strictly greater than A_i (which is x).
            # This requires that all fixed B_k (k < i-1) must be greater than x.
            if m_fixed > x:
                # If conditions are met, each of the q_prev -1s in B[0...i-2] can be filled
                # with any value from (x+1) to M, giving (M-x) choices.
                term_i = pows[M-x][q_prev]
            else:
                term_i = 0 # An invalid configuration exists (a fixed B_k <= x).
        else: # Case 2: B[i-1] is -1. A_i can be any value `x` from 1 to M.
            # We sum the possibilities for each choice of `x`.
            # For a given `x`, the count is (M-x)^q_prev if m_fixed > x, else 0.
            # The sum is for `x` from 1 up to `min(M, m_fixed-1)`.
            # `min(M + 1, m_fixed)` handles the upper bound correctly for `range()`.
            # If `m_fixed` is 1, `range(1, 1)` is empty, `term_i` remains 0 (correct).
            for x in range(1, min(M + 1, m_fixed)):
                term_i = (term_i + pows[M-x][q_prev]) % MOD
        
        # q_suffix_i: number of -1s in B[i...N-1].
        # These positions can be filled arbitrarily with M choices each.
        q_suffix_i = total_q - num_neg1_prefix[i]
        
        # Add contribution of vertex `i` being a root to the total sum.
        # This is `term_i` multiplied by M^(q_suffix_i).
        ans = (ans + term_i * pows[M][q_suffix_i]) % MOD

    sys.stdout.write(str(ans) + "
")

solve()