import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    MOD = 998244353

    conditions_raw = []
    for _ in range(M):
        L, R, X = map(int, sys.stdin.readline().split())
        # If L=R, then by constraints L=X=R. This means P_X must not be max in {P_X}.
        # P_X is always max in {P_X}, so P_X != P_X is the condition, which is impossible.
        if L == R: 
             print(0)
             return
        conditions_raw.append((L, R, X))

    MAX_N_COMB = N + 5 
    fact = [1] * MAX_N_COMB
    inv_fact = [1] * MAX_N_COMB
    for i in range(1, MAX_N_COMB):
        fact[i] = (fact[i-1] * i) % MOD
    
    inv_fact[MAX_N_COMB-1] = pow(fact[MAX_N_COMB-1], MOD-2, MOD)
    for i in range(MAX_N_COMB-2, -1, -1): 
        inv_fact[i] = (inv_fact[i+1] * (i+1)) % MOD

    def nCr_mod(n, r):
        if r < 0 or r > n:
            return 0
        num = fact[n]
        den = (inv_fact[r] * inv_fact[n-r]) % MOD
        return (num * den) % MOD

    GlobalSuffMinR = [[N + 1] * (N + 2) for _ in range(N + 1)]

    conditions_by_X = [[] for _ in range(N + 1)]
    for L, R, X in conditions_raw:
        conditions_by_X[X].append((L, R))

    for piv in range(1, N + 1):
        tmp_actual_min_R = [N + 1] * (piv + 1) 
        for L_s, R_s in conditions_by_X[piv]:
            # L_s <= X_s (=piv) must hold due to problem constraints
            if L_s <= piv: # Defensive check, L_s guaranteed to be <= piv by problem constraints
                 tmp_actual_min_R[L_s] = min(tmp_actual_min_R[L_s], R_s)
        
        # GlobalSuffMinR[piv][piv+1] is already N+1 (sentinel for suffix minimum calculation)
        for L_val in range(piv, 0, -1): # L_val from piv down to 1
            GlobalSuffMinR[piv][L_val] = min(tmp_actual_min_R[L_val], GlobalSuffMinR[piv][L_val+1])

    dp = [[0] * (N + 2) for _ in range(N + 1)] 

    for i in range(1, N + 2): 
        dp[0][i] = 1
    
    for length in range(1, N + 1):
        for i in range(1, N - length + 2): 
            j = i + length - 1 
            
            current_dp_val = 0
            for piv_offset in range(length): 
                piv = i + piv_offset 
                
                is_bad = (GlobalSuffMinR[piv][i] <= j)
                
                if is_bad:
                    continue
                
                left_len = piv_offset
                right_len = length - 1 - piv_offset
                
                combinations = nCr_mod(length - 1, left_len)
                
                term = combinations
                term = (term * dp[left_len][i]) % MOD
                term = (term * dp[right_len][piv + 1]) % MOD
                
                current_dp_val = (current_dp_val + term) % MOD
            
            dp[length][i] = current_dp_val
            
    print(dp[N][1])

solve()