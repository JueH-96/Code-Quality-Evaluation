import sys

def get_v2(n):
    # Returns exponent of 2 in prime factorization of n
    if n == 0: # Should not happen for hook lengths
        return float('inf') 
    if n < 0: # Should not happen
        n = -n
    
    count = 0
    while n > 0 and n % 2 == 0:
        count += 1
        n //= 2
    return count

def get_v2_factorial(k):
    # Returns exponent of 2 in prime factorization of k!
    if k < 0: return 0 # Or error
    if k == 0: return 0
    
    count = 0
    power_of_2 = 2
    while power_of_2 <= k:
        count += k // power_of_2
        if power_of_2 > k // 2: # Optimization: k // (power_of_2 * 2) would be 0
            break
        power_of_2 *= 2
    return count

def solve():
    A, B, M = map(int, sys.stdin.readline().split())

    N = A * B - 1

    # Shape lambda_0: B-1 rows of length A, 1 row of length A-1
    # lambda_rows[i] is length of (i+1)-th row
    lambda_rows = [A] * (B - 1) + [A - 1]
    
    # lambda_cols[j] is length of (j+1)-th col
    # A-1 cols of length B, 1 col of length B-1
    lambda_cols = [B] * (A - 1) + [B - 1]

    # Calculate N! mod M
    fact_N_mod_M = 1
    for i in range(1, N + 1):
        fact_N_mod_M = (fact_N_mod_M * i) % M

    # Modular exponentiation for inverse
    def power(a, b_exp):
        res = 1
        a %= M
        while b_exp > 0:
            if b_exp % 2 == 1:
                res = (res * a) % M
            a = (a * a) % M
            b_exp //= 2
        return res

    def inv_mod_M(n_mod_M):
        return power(n_mod_M, M - 2)

    # Calculate product of hook lengths mod M and sum of v2(hook_lengths)
    prod_hooks_mod_M = 1
    sum_v2_hooks = 0
    
    # Iterate over cells (r, c) of the Young diagram (1-indexed)
    for r in range(1, B + 1): # r from 1 to B
        current_row_len = lambda_rows[r-1]
        for c in range(1, current_row_len + 1): # c from 1 to lambda_rows[r-1]
            # Hook length h_rc = (lambda_r - c) + (lambda'_c - r) + 1
            hook_len_val = (lambda_rows[r-1] - c) + (lambda_cols[c-1] - r) + 1
            
            prod_hooks_mod_M = (prod_hooks_mod_M * hook_len_val) % M
            sum_v2_hooks += get_v2(hook_len_val)
            
    # f_lambda0 mod M
    f_lambda0_mod_M = (fact_N_mod_M * inv_mod_M(prod_hooks_mod_M)) % M
    
    # Parity of f_lambda0
    v2_fact_N = get_v2_factorial(N)
    v2_f_lambda0 = v2_fact_N - sum_v2_hooks # This is v_2(f_lambda0)

    # N_S_lambda0 = floor(f_lambda0 / 2)
    # Calculate N_S_lambda0 mod M
    
    N_S_lambda0_mod_M = 0
    inv2_mod_M = inv_mod_M(2)

    if v2_f_lambda0 > 0: # f_lambda0 is even
        N_S_lambda0_mod_M = (f_lambda0_mod_M * inv2_mod_M) % M
    else: # f_lambda0 is odd
        # (f_lambda0_mod_M - 1)/2 mod M
        term_val = (f_lambda0_mod_M - 1 + M) % M 
        N_S_lambda0_mod_M = (term_val * inv2_mod_M) % M
            
    ans = (N_S_lambda0_mod_M * f_lambda0_mod_M) % M
    
    sys.stdout.write(str(ans) + "
")

solve()