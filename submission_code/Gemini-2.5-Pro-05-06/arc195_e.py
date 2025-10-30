import sys

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    
    A_values_from_input = list(map(int, sys.stdin.readline().split()))
    
    MOD = 998244353

    A = [0] * (N + 1) 
    # A_values_from_input are A_2, ..., A_N. A_values_from_input[i] is A_{i+2}.
    for i in range(N - 1): # N-1 elements for A_2 to A_N
        A[i+2] = A_values_from_input[i]

    inv = [1] * (N + 1)
    if N >= 1:
        inv[1] = 1
    for i in range(2, N + 1):
        inv[i] = MOD - (MOD // i) * inv[MOD % i] % MOD
    
    X = [0] * (N + 1) # X_1 = 0
    
    # Pref_term_val[k] = sum_{j=2}^k (A_j / j)
    Pref_term_val = [0] * (N + 1) # Pref_term_val[0,1] = 0
    if N >= 2:
        for k in range(2, N + 1):
            term_k_val = A[k] * inv[k] % MOD
            Pref_term_val[k] = (Pref_term_val[k-1] + term_k_val) % MOD
            
    if N >= 2:
        for i in range(2, N + 1):
            X[i] = A[i]
            # Sum A_j/j for j from 2 to i-1. This is Pref_term_val[i-1].
            # If i-1 < 2 (i.e. i < 3), this sum is empty (0). Pref_term_val[0 or 1] is 0.
            X[i] = (X[i] + Pref_term_val[i-1]) % MOD
    
    # S_X[k] = sum_{r=2}^k X_r
    S_X = [0] * (N + 1) # S_X[0,1] = 0
    if N >= 2:
        for k in range(2, N + 1):
            S_X[k] = (S_X[k-1] + X[k]) % MOD

    fact_N_minus_1 = 1
    for i in range(1, N): # computes (N-1)!
        fact_N_minus_1 = fact_N_minus_1 * i % MOD
    
    results = []
    for _ in range(Q):
        u, v = map(int, sys.stdin.readline().split())
        if u > v:
            u, v = v, u 

        if u == 1:
            ans_exp = X[v]
        else: # u > 1
            Y_uv_term1 = X[u] * inv[u] % MOD
            
            Y_uv_term2 = 0
            # S_X[u-1] is sum X_r for r=2 to u-1. Correctly 0 if u-1 < 2 via S_X[0 or 1].
            sum_val_for_term2 = S_X[u-1] 
            if sum_val_for_term2 != 0: # Denominator (u-1)(v-1)
                den_inv = inv[u-1] * inv[v-1] % MOD
                Y_uv_term2 = sum_val_for_term2 * den_inv % MOD
            
            Y_uv = (Y_uv_term1 + Y_uv_term2) % MOD
            
            ans_exp = (X[u] + X[v] - (2 * Y_uv % MOD) + MOD) % MOD
            
        final_ans = ans_exp * fact_N_minus_1 % MOD
        results.append(str(final_ans))

    sys.stdout.write('
'.join(results) + '
')

solve()