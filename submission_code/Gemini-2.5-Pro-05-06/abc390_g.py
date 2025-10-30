import sys

# It's good practice for competitive programming but not strictly needed here
# sys.setrecursionlimit(4 * 10**5) 

def solve():
    N = int(sys.stdin.readline())
    MOD = 998244353

    if N == 0: 
        print(0)
        return
    # N=1 is handled by general logic, f((1))=1.
    # Example, N=1. Nd[1]=1, Sd[1]=1. GlobalP=(1+10y)^1. G_L_coeffs=[1,10]. Trim to N=1 -> [1].
    # d0=1. C_0=G_0=1. C_L_coeffs=[1].
    # SumForD0: L=0: (1-0-1)!*0!*C_0 = 0!0!*1 = 1.
    # TotalAns = Sd[1]*1 = 1*1=1. Correct.

    fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = (fact[i-1] * i) % MOD

    pow10_val = [1] * 7 
    for i in range(1, 7):
        pow10_val[i] = pow10_val[i-1] * 10 # Actual powers, not mod yet

    N_d = [0] * 7
    S_d = [0] * 7
    current_lower_bound = 1
    for d_digits in range(1, 7):
        upper_bound_for_d_digits = pow10_val[d_digits] - 1
        actual_upper = min(N, upper_bound_for_d_digits)
        actual_lower = current_lower_bound
        
        if actual_upper >= actual_lower:
            N_d[d_digits] = actual_upper - actual_lower + 1
            current_S_d_val = (N_d[d_digits] * (actual_lower + actual_upper)) // 2
            S_d[d_digits] = current_S_d_val % MOD
            
        if N <= upper_bound_for_d_digits:
            break
        current_lower_bound = upper_bound_for_d_digits + 1

    ROOT_PRIMITIVE = 3
    def ntt(P, invert): # Standard iterative NTT
        n = len(P)
        for i in range(1, n): # Bit-reversal permutation
            j = 0; bit = n >> 1; temp_i = i
            while bit > 0:
                if temp_i & 1: j |= bit
                temp_i >>= 1; bit >>= 1
            if i < j: P[i], P[j] = P[j], P[i]
        
        len_ = 2
        while len_ <= n:
            w_len = pow(ROOT_PRIMITIVE, (MOD - 1) // len_, MOD)
            if invert: w_len = pow(w_len, MOD - 2, MOD)
            i = 0
            while i < n:
                w = 1
                for k in range(len_ // 2):
                    u, v = P[i+k], (P[i+k+len_//2] * w) % MOD
                    P[i+k], P[i+k+len_//2] = (u+v)%MOD, (u-v+MOD)%MOD
                    w = (w * w_len) % MOD
                i += len_
            len_ <<= 1
        if invert:
            n_inv = pow(n, MOD-2, MOD)
            for i in range(n): P[i] = (P[i] * n_inv) % MOD
        return P

    def multiply(A_coeffs, B_coeffs): # Multiply two polynomials
        len_A, len_B = len(A_coeffs), len(B_coeffs)
        if len_A == 0 or len_B == 0: return []
        # Optimization for multiplication by [0] or [1] can be added if needed
        # e.g. if A_coeffs == [1], return B_coeffs
        
        res_len = len_A + len_B - 1
        n_ntt = 1
        while n_ntt < res_len: n_ntt <<= 1
        
        A_ntt = A_coeffs + [0]*(n_ntt-len_A); ntt(A_ntt, False)
        B_ntt = B_coeffs + [0]*(n_ntt-len_B); ntt(B_ntt, False)
        C_ntt = [(a*b)%MOD for a,b in zip(A_ntt, B_ntt)]; ntt(C_ntt, True)
        return C_ntt[:res_len]

    binom_coeffs_cache = {}
    def nCr_mod(n_val, r_val): # nCr with caching
        state = (n_val, r_val)
        if state in binom_coeffs_cache: return binom_coeffs_cache[state]
        if r_val < 0 or r_val > n_val: return 0
        num = fact[n_val]
        den = (fact[r_val] * fact[n_val-r_val]) % MOD
        res = (num * pow(den, MOD-2, MOD)) % MOD
        binom_coeffs_cache[state] = res
        return res

    GlobalP_coeffs = [1]
    for d_idx in range(1, 7):
        if N_d[d_idx] == 0: continue
        Q_j_len = N_d[d_idx] + 1
        Q_j_coeffs = [0] * Q_j_len
        term_A_coeff = pow10_val[d_idx] % MOD
        for r_power in range(N_d[d_idx] + 1):
            binom_val = nCr_mod(N_d[d_idx], r_power)
            term_A_pow_r = pow(term_A_coeff, r_power, MOD)
            Q_j_coeffs[r_power] = (binom_val * term_A_pow_r) % MOD
        GlobalP_coeffs = multiply(GlobalP_coeffs, Q_j_coeffs)
        if len(GlobalP_coeffs) > N: GlobalP_coeffs = GlobalP_coeffs[:N]

    if len(GlobalP_coeffs) < N:
        GlobalP_coeffs.extend([0] * (N - len(GlobalP_coeffs)))
    G_L_coeffs = GlobalP_coeffs

    total_ans = 0
    for d0_X_digits in range(1, 7):
        if N_d[d0_X_digits] == 0: continue
        C_L_coeffs = [0] * N
        term_A_for_d0 = pow10_val[d0_X_digits] % MOD
        
        if N > 0: # Ensure G_L_coeffs is not empty (which it isn't if N > 0)
            C_L_coeffs[0] = G_L_coeffs[0] 
            for L_idx in range(1, N):
                term_to_subtract = (term_A_for_d0 * C_L_coeffs[L_idx-1]) % MOD
                C_L_coeffs[L_idx] = (G_L_coeffs[L_idx] - term_to_subtract + MOD) % MOD
        
        current_sum_for_this_d0_type = 0
        for L_idx in range(N):
            term_factorials = (fact[N-L_idx-1] * fact[L_idx]) % MOD
            term = (term_factorials * C_L_coeffs[L_idx]) % MOD
            current_sum_for_this_d0_type = (current_sum_for_this_d0_type + term) % MOD
            
        term_S_d0 = S_d[d0_X_digits]
        total_ans = (total_ans + term_S_d0 * current_sum_for_this_d0_type) % MOD

    print(total_ans)

solve()