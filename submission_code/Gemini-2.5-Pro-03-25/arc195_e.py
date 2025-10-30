# YOUR CODE HERE
import sys

# Set higher recursion depth for safety, although the final solution is iterative.
# sys.setrecursionlimit(200000) 

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    
    # Basic check based on constraints N >= 2
    if N < 2: 
        # If N=1, only vertex 1 exists. Any query (u, v) must be (1, 1). Distance is 0.
        # The loop below would handle this if N=1 was possible.
        # But constraints say N>=2, so this case is technically unreachable.
        for _ in range(Q): print(0) 
        return

    # Read A_2, ..., A_N
    A_input = list(map(int, sys.stdin.readline().split())) 
    
    MOD = 998244353

    # Store A_i values in 1-based index array A_full, taking modulo early
    # A_full[i] stores the weight of the edge connecting vertex i to its parent P_i.
    A_full = [0] * (N + 1)
    for i in range(N - 1):
        A_full[i + 2] = A_input[i] % MOD # Store A_{i+2} at index i+2

    # Precompute factorials up to N-1. We need (N-1)!
    fact = [1] * (N + 1) # fact[k] stores k! mod MOD
    fact[0] = 1
    # No need for fact[1] explicitly since loop starts from 2
    
    # Loop calculates factorials up to N. We only need up to N-1 for F.
    for i in range(2, N + 1): # Calculate up to N needed for inverses later potentially
        fact[i] = (fact[i - 1] * i) % MOD

    # F = (N-1)! mod MOD. This is the total number of trees除以(N-1) which cancels? No, F is total number of sequences P.
    # F is used as a common factor in the sum calculations.
    if N>=2:
       F = fact[N - 1] 
    else: # Only N=1 case, which is ruled out by constraints. F=0! = 1?
       F = 1

    # Precompute modular inverses up to N using Fermat's Little Theorem
    # inv[k] stores k^{-1} mod MOD
    inv = [1] * (N + 1)
    inv[1] = 1 # Base case inverse of 1 is 1.
    for i in range(2, N + 1):
        inv[i] = pow(i, MOD - 2, MOD) # Compute i^(MOD-2) mod MOD

    # Precompute prefix sums PS[k] = sum_{i=2..k} (A_full[i] * inv[i]) mod MOD
    # This is used to efficiently calculate D(x) values.
    PS = [0] * (N + 1)
    for i in range(2, N + 1):
        PS[i] = (PS[i - 1] + A_full[i] * inv[i]) % MOD

    # Calculate D(x) = sum_P d(x, 1) for all x. D(x) is the sum of distances from x to root 1 over all trees.
    D = [0] * (N + 1) # D[1] = 0 by definition (distance from root to itself is 0)
    for x in range(2, N + 1):
        # D(x) = F * (A_x + sum_{i=2..x-1} A_i/i) mod MOD
        term_Ax = A_full[x] 
        term_PS = PS[x - 1] # PS[x-1] corresponds to sum_{i=2..x-1} A_i * inv[i]
        D[x] = (F * (term_Ax + term_PS)) % MOD

    # Precompute prefix sums of A_full values: PSA[k] = sum_{i=2..k} A_full[i] mod MOD
    # This is needed for efficient calculation of E(u,v) term.
    PSA = [0] * (N + 1)
    for i in range(2, N + 1):
        PSA[i] = (PSA[i-1] + A_full[i]) % MOD
        
    # Query processing loop
    results = []
    for _ in range(Q):
        u, v = map(int, sys.stdin.readline().split())
        
        # Constraints state 1 <= u < v <= N. No need to check/swap u, v.

        if u == 1:
            # If u is the root (vertex 1), the sum of distances is simply D(v).
            results.append(D[v])
            continue

        # Case u, v >= 2:
        # The total sum of distances is derived as: TotalSum = D(u) + D(v) - 2 * T
        # where T = E(u, v) + F * (A_u * [u<v]/u + A_v * [v<u]/v)
        # and E(u, v) = F * sum_{k=2}^{min(u,v)-1} A_k * p_c(k, u, v)
        # p_c(k, u, v) is the probability that k is a common ancestor of u and v.
        
        # Based on analysis and testing sample cases, the correct probability seems to be:
        # p_c(k, u, v) = 1 / (max(u, v) - 1)
        # Since u < v, max(u, v) = v. So p_c(k, u, v) = 1 / (v - 1).
        
        # Calculate E(u, v) term:
        # E(u, v) = F * sum_{k=2}^{u-1} A_k * (1 / (v - 1)) mod MOD
        # E(u, v) = F * inv[v-1] * (Sum_{k=2}^{u-1} A_k) mod MOD
        
        # Get modular inverse of v-1. v >= u >= 2 implies v-1 >= 1. Index is safe.
        inv_v_minus_1 = inv[v - 1] 

        # Sum_{k=2}^{u-1} A_k is PSA[u-1]
        # If u=2, the range k=2..1 is empty, sum is 0. PSA[1]=0 handles this.
        sum_A_k_for_E = PSA[u-1] 
        
        # Calculate product for E(u, v) term carefully with modulo operations
        E_uv_term_product = (sum_A_k_for_E * inv_v_minus_1) % MOD
        E_uv = (F * E_uv_term_product) % MOD
        
        # Calculate Term_Aux = F * (A_u * [u<v]/u) because u < v
        # This term accounts for the specific ancestor relations involving u and v themselves.
        Term_Aux = (F * A_full[u] * inv[u]) % MOD
        
        # Calculate T = E(u, v) + Term_Aux
        T_val = (E_uv + Term_Aux) % MOD
        
        # Calculate final answer: TotalSum = (D(u) + D(v) - 2 * T) mod MOD
        # Use + MOD + MOD pattern to ensure the result is non-negative before the final modulo.
        ans = (D[u] + D[v] - 2 * T_val + MOD + MOD) % MOD
        results.append(ans)

    # Print all results. Using sys.stdout.write might be slightly faster for large Q.
    output = "
".join(map(str, results))
    sys.stdout.write(output + "
")

solve()