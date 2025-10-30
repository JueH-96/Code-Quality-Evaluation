# YOUR CODE HERE
import sys

# Setting a reasonable recursion depth, although the solution is iterative.
# sys.setrecursionlimit(2010) 

def solve():
    N, M = map(int, sys.stdin.readline().split())
    B = list(map(int, sys.stdin.readline().split()))
    MOD = 998244353

    # Count the total number of -1s in the sequence B
    q = B.count(-1)

    # Precompute powers of M modulo MOD. We need powers up to q.
    # Since q can be up to N, precomputing up to N is safe.
    M_pow = [1] * (N + 1) 
    for i in range(1, N + 1):
        M_pow[i] = (M_pow[i-1] * M) % MOD

    # Modular exponentiation function: calculates (a^b) % MOD
    def power(a, b):
        # Handle edge cases for base 0 or 1 efficiently
        if a == 0: return 0 if b > 0 else 1
        if a == 1: return 1
        
        res = 1
        a %= MOD
        # Python's % operator handles negative results correctly for positive modulus,
        # but explicit check `if a < 0: a += MOD` might be needed in other languages.
        
        while b > 0:
            if b % 2 == 1:
                res = (res * a) % MOD
            a = (a * a) % MOD
            # Optimization: if a becomes 0, future multiplications yield 0.
            if a == 0 and b > 1: 
                 # This check avoids unnecessary loops if a becomes 0.
                 # The check `res == 0` handles the case where res might have already become 0.
                 return 0 if res == 0 else res 
            b //= 2
        return res

    # Compute prefix counts of -1s
    # q_prefix[k] = number of -1s in B[0...k-1]
    q_prefix = [0] * (N + 1)
    for k in range(N):
        q_prefix[k+1] = q_prefix[k] + (1 if B[k] == -1 else 0)

    # Compute suffix counts of -1s
    # q_suffix[k] = number of -1s in B[k...N-1]
    q_suffix = [0] * (N + 1) 
    for k in range(N - 1, -1, -1):
       q_suffix[k] = q_suffix[k+1] + (1 if B[k] == -1 else 0)
       
    # Initialize the total sum of f(B') over all possible B'
    total_sum_f = 0

    # Calculate N_1, the contribution of vertex 1.
    # L(1)=1 is always true for any sequence A.
    # The number of sequences B' where L(1)=1 is the total number of sequences, M^q.
    if q >= 0 : # q is always non-negative
      # Check array bounds just in case, although N >= 2 implies q could be >= 0
      if q < len(M_pow):
           N_1 = M_pow[q] 
      else: # Fallback calculation if q somehow exceeds N (should not happen)
           N_1 = power(M, q) 
    else: 
      N_1 = 1 # M^0 = 1 if q=0
    
    total_sum_f = (total_sum_f + N_1) % MOD
    
    # Keep track of the minimum value among fixed B_j for j < i (0-based index)
    # Initialize with M+1 ensures any valid value 1..M will be smaller.
    min_fixed_val_prefix = M + 1 

    # Loop through vertices i = 2 to N. The 0-based index is i-1.
    # In each iteration `i`, we calculate N_i, the contribution of vertex `i` (1-based).
    # The loop variable `current_vertex_idx_0based` represents the 0-based index corresponding to vertex i.
    for current_vertex_idx_0based in range(1, N): 
        
        # Update min_fixed_val_prefix using the value at the previous vertex
        # B[current_vertex_idx_0based - 1] corresponds to vertex i-1 (1-based)
        prev_idx = current_vertex_idx_0based - 1
        if B[prev_idx] != -1:
             min_fixed_val_prefix = min(min_fixed_val_prefix, B[prev_idx])

        # Calculate N_i for vertex with 0-based index `current_vertex_idx_0based` (vertex i+1, 1-based)
        # q_lt_i is the count of -1s in B[0...i-1]
        q_lt_i = q_prefix[current_vertex_idx_0based] 
        # q_gt_i is the count of -1s in B[i+1...N-1]
        q_gt_i = q_suffix[current_vertex_idx_0based + 1] 
        
        # m_i is the minimum fixed value among vertices before the current one (vertex i+1)
        m_i = min_fixed_val_prefix 

        # Initialize contribution of vertex i+1 (1-based)
        N_i_val = 0 

        # Check if the value A_i is fixed (B[i] != -1)
        if B[current_vertex_idx_0based] != -1: 
             
             fixed_Ai = B[current_vertex_idx_0based]
             
             # Condition for L(i)=i is: for all j < i, A_j > A_i.
             # For fixed B_j (j<i), need B_j > fixed_Ai. This implies m_i > fixed_Ai.
             if m_i > fixed_Ai:
                 # For j < i where B_j = -1, need to choose A_j > fixed_Ai.
                 # Number of choices is M - fixed_Ai.
                 # There are q_lt_i such positions. Total ways: (M - fixed_Ai) ^ q_lt_i.
                 
                 # Base for exponentiation must be non-negative
                 term_power_base = M - fixed_Ai
                 # This case should not happen if M >= 1 and fixed_Ai <= M,
                 # unless M < fixed_Ai, but that's covered by m_i > fixed_Ai.
                 # If m_i > fixed_Ai, then fixed_Ai < m_i <= M. So M - fixed_Ai >= 0.
                 
                 term = power(term_power_base, q_lt_i)
                 
                 # Multiply by ways to fill positions > i. There are q_gt_i such positions.
                 # Each can be filled in M ways. Total ways M^{q_gt_i}.
                 if q_gt_i < len(M_pow):
                      N_i_val = (term * M_pow[q_gt_i]) % MOD
                 else: # Fallback: Should not happen with M_pow size N+1
                      N_i_val = (term * power(M, q_gt_i)) % MOD
             # If m_i <= fixed_Ai, the condition fails for some fixed B_j, so N_i_val remains 0.

        # Case 2: The value A_i is variable (B[i] == -1).
        else: 
            
            # A_i can be assigned any value v from 1 to M.
            # Condition L(i)=i requires: for all j < i, A_j > A_i=v.
            # For fixed B_j (j<i), need B_j > v. This requires m_i > v.
            # So, possible values for v are 1, 2, ..., m_i - 1. The maximum value for v is M.
            
            sum_terms = 0
            # Effective upper limit (exclusive) for v is min(M + 1, m_i).
            # Loop iterates v from 1 up to limit_v - 1.
            limit_v = min(M + 1, m_i) 
            
            # Sum over possible values v for A_i
            for v in range(1, limit_v):
                 # For a fixed v, we need to assign values for j < i where B_j = -1.
                 # For such j, need A_j > v. Number of choices is M - v.
                 # There are q_lt_i such positions. Total ways is (M - v)^q_lt_i.
                 term_power_base = M - v
                 # Since v >= 1 and v <= M, M-v >= 0.
                 
                 term = power(term_power_base, q_lt_i)
                 sum_terms = (sum_terms + term) % MOD
            
            # Multiply the sum of ways for A_1..A_i by the number of ways to fill positions > i.
            if q_gt_i < len(M_pow):
                 N_i_val = (sum_terms * M_pow[q_gt_i]) % MOD
            else: # Fallback: Should not happen
                 N_i_val = (sum_terms * power(M, q_gt_i)) % MOD

        # Add the contribution N_i_val for the current vertex to the total sum
        total_sum_f = (total_sum_f + N_i_val) % MOD

    # Print the final computed sum modulo MOD
    print(total_sum_f)

# Execute the solve function
solve()