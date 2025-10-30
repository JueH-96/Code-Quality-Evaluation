import sys

def solve():
    N, M_len_divisor, K_power = map(int, sys.stdin.readline().split())
    A_list = list(map(int, sys.stdin.readline().split()))

    MOD = 998244353

    fixed_L = 20 
    size = 1 << fixed_L

    freq = [0] * size
    for x_val in A_list:
        freq[x_val] += 1

    # WHT (FWHT)
    # Transforms f to f_hat[s] = sum_x f[x] * (-1)**(s.x)
    # W_freq values will be used to calculate N_s_plus and N_s_minus.
    # These calculations need values mod MOD that correspond to true sums S_s mod MOD.
    # Standard WHT applies additions/subtractions. Python integers handle arbitrary size.
    # If MOD is applied at each step, it's correct.
    w_freq = list(freq) 
    for i in range(fixed_L): 
        for mask in range(size):
            if not (mask & (1 << i)): # if i-th bit of mask is 0
                u = w_freq[mask]
                v = w_freq[mask | (1 << i)]
                w_freq[mask] = (u + v) % MOD
                w_freq[mask | (1 << i)] = (u - v + MOD) % MOD
    
    P_poly_coeffs = [[0] * M_len_divisor for _ in range(N + 1)]
    M_poly_coeffs = [[0] * M_len_divisor for _ in range(N + 1)]

    P_poly_coeffs[0][0] = 1
    M_poly_coeffs[0][0] = 1

    for k in range(N):
        # P_poly_coeffs[k+1] from P_poly_coeffs[k] * (1+omega)
        pk_curr = P_poly_coeffs[k]
        pk_next = P_poly_coeffs[k+1]
        for j in range(M_len_divisor):
            pk_next[j] = (pk_curr[j] + pk_curr[(j - 1 + M_len_divisor) % M_len_divisor]) % MOD
        
        # M_poly_coeffs[k+1] from M_poly_coeffs[k] * (1-omega)
        mk_curr = M_poly_coeffs[k]
        mk_next = M_poly_coeffs[k+1]
        for j in range(M_len_divisor):
            mk_next[j] = (mk_curr[j] - mk_curr[(j - 1 + M_len_divisor) % M_len_divisor] + MOD) % MOD

    C_s_array = [0] * size 
    inv2 = pow(2, MOD - 2, MOD)

    for s in range(size):
        # N_s_plus and N_s_minus are integer counts.
        # w_freq[s] is S_s mod MOD. (N + S_s mod MOD) * inv2 mod MOD gives ( (N+S_s)/2 ) mod MOD.
        # This result, cast to int, is the correct count N_s_plus, as it's in [0, N].
        idx_A = int((N + w_freq[s]) * inv2 % MOD)
        idx_B = int((N - w_freq[s] + MOD) * inv2 % MOD)
        
        current_C_s = 0
        # Coeffs for (1+omega)^idx_A
        ptr_P_coeffs = P_poly_coeffs[idx_A]
        # Coeffs for (1-omega)^idx_B
        ptr_M_coeffs = M_poly_coeffs[idx_B]

        # Convolution for C_s = coeff_0 of (P_A * M_B)
        # C_s = sum_{j=0 to M-1} P_A[j] * M_B[(M-j)%M]
        current_C_s = ptr_P_coeffs[0] * ptr_M_coeffs[0] % MOD 
        for j_term in range(1, M_len_divisor): 
            current_C_s = (current_C_s + ptr_P_coeffs[j_term] * ptr_M_coeffs[M_len_divisor-j_term]) % MOD
        
        C_s_array[s] = current_C_s

    # IWHT for C_s_array to get dp_counts_len_mod_0[X]
    # dp_counts_len_mod_0[X] = (1/size) * sum_s C_s_array[s] * (-1)**(s.X)
    dp_counts_len_mod_0 = C_s_array # Modify in place
    for i in range(fixed_L): 
        for mask in range(size):
            if not (mask & (1 << i)): 
                u = dp_counts_len_mod_0[mask]
                v = dp_counts_len_mod_0[mask | (1 << i)]
                dp_counts_len_mod_0[mask] = (u + v) % MOD
                dp_counts_len_mod_0[mask | (1 << i)] = (u - v + MOD) % MOD
    
    inv_size = pow(size, MOD - 2, MOD)
    for i in range(size):
        dp_counts_len_mod_0[i] = dp_counts_len_mod_0[i] * inv_size % MOD
            
    total_score = 0
    # Sum (dp_counts_len_mod_0[X]) * X^K_power for X > 0
    # (dp_counts_len_mod_0[0]-1) * 0^K_power for X = 0. Since K_power >= 1, 0^K_power = 0.
    # So the X=0 term contributes 0 to the sum of scores.
    for x_val in range(1, size): # Iterate X from 1 up to size-1
        if dp_counts_len_mod_0[x_val] == 0:
            continue
        
        # Compute x_val^K_power % MOD
        # pow() is efficient for this.
        term_score_val = pow(x_val, K_power, MOD)
        
        total_score = (total_score + dp_counts_len_mod_0[x_val] * term_score_val) % MOD
        
    print(total_score)

solve()