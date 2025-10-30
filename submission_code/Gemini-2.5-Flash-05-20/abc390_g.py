import sys

def solve():
    N = int(sys.stdin.readline())

    MOD = 998244353
    INV2 = pow(2, MOD - 2, MOD) # Precompute inverse of 2 for arithmetic series sum

    # Precompute factorials and inverse factorials
    # Max factorial needed is N-1. So array up to N.
    MAX_FAC = N
    fact = [1] * (MAX_FAC + 1)
    inv_fact = [1] * (MAX_FAC + 1)
    for i in range(1, MAX_FAC + 1):
        fact[i] = (fact[i-1] * i) % MOD
    inv_fact[MAX_FAC] = pow(fact[MAX_FAC], MOD - 2, MOD)
    for i in range(MAX_FAC - 1, -1, -1):
        inv_fact[i] = (inv_fact[i+1] * (i+1)) % MOD

    # Precompute powers of 10
    # Max exponent needed is max_L * (N-1). For N=2e5, max_L=6, this is ~1.2e6.
    MAX_POW10_EXP = 6 * N 
    pow10 = [1] * (MAX_POW10_EXP + 1)
    for i in range(1, MAX_POW10_EXP + 1):
        pow10[i] = (pow10[i-1] * 10) % MOD

    # Determine number of digits for x, and precompute counts and sum_val
    # max_L is max digits for any number from 1 to N
    max_L = len(str(N)) if N > 0 else 1
    
    counts = [0] * (max_L + 1) # counts[d] = number of d-digit numbers <= N
    sum_val = [0] * (max_L + 1) # sum_val[d] = sum of d-digit numbers <= N

    current_power_of_10 = 1
    for d in range(1, max_L + 1):
        start = current_power_of_10
        end = min(N, current_power_of_10 * 10 - 1)
        
        if start > N: # No numbers with d digits up to N
            break
        
        count_d = end - start + 1
        
        # Sum of arithmetic series: (first + last) * count / 2
        sum_d_arithmetic_series = (start % MOD + end % MOD) % MOD
        sum_d_arithmetic_series = (sum_d_arithmetic_series * (count_d % MOD)) % MOD
        sum_d_arithmetic_series = (sum_d_arithmetic_series * INV2) % MOD
        
        counts[d] = count_d
        sum_val[d] = sum_d_arithmetic_series
        
        current_power_of_10 *= 10
    
    # 2. Compute dp[j] (coefficients c_j for P(z) = product (1 + z * 10^d)^counts[d])
    # P(z) has degree N. dp[j] stores c_j.
    dp = [0] * (N + 1)
    dp[0] = 1
    
    current_max_dp_degree = 0
    for d in range(1, max_L + 1):
        if counts[d] == 0:
            continue
        term_val = pow10[d] # This is 10^d
        
        # Apply (1 + z*term_val) 'counts[d]' times
        # Each application increases degree by 1.
        for _ in range(counts[d]):
            # Iterate j downwards to use previous dp values before they're updated
            for j in range(current_max_dp_degree, -1, -1):
                if j + 1 <= N: # Check bounds for dp[j+1]
                    dp[j+1] = (dp[j+1] + dp[j] * term_val) % MOD
            current_max_dp_degree = min(N, current_max_dp_degree + 1) # Cap at N because dp represents P(z) of degree N

    # 3. Compute pow_val_sum[p] = Sum_{k=1}^N k * (10^{L(k)})^p
    # p goes from 0 to N-1
    pow_val_sum = [0] * N 
    for p in range(N):
        current_p_sum = 0
        for d in range(1, max_L + 1):
            if counts[d] == 0:
                continue
            
            # (10^d)^p = 10^(d*p)
            pow_10_dp = pow10[d * p]
            current_p_sum = (current_p_sum + pow_10_dp * sum_val[d]) % MOD
        pow_val_sum[p] = current_p_sum

    # 4. Compute B_m = (-1)^m * pow_val_sum[m]
    B = [0] * N
    for m in range(N):
        if m % 2 == 0:
            B[m] = pow_val_sum[m]
        else:
            B[m] = (MOD - pow_val_sum[m]) % MOD

    # 5. Compute C = dp * B using NTT
    PRIMITIVE_ROOT = 3
    
    def ntt(a, invert):
        n = len(a)
        j = 0
        for i in range(1, n):
            if i < j:
                a[i], a[j] = a[j], a[i]
            k = n // 2
            while k <= j:
                j -= k
                k //= 2
            j += k

        len_ntt = 2
        while len_ntt <= n:
            wlen = pow(PRIMITIVE_ROOT, (MOD - 1) // len_ntt, MOD)
            if invert:
                wlen = pow(wlen, MOD - 2, MOD)
            
            for i in range(0, n, len_ntt):
                w = 1
                for k_inner in range(len_ntt // 2): # Renamed j to k_inner to avoid conflict with outer j
                    u = a[i + k_inner]
                    v = (a[i + k_inner + len_ntt // 2] * w) % MOD
                    a[i + k_inner] = (u + v) % MOD
                    a[i + k_inner + len_ntt // 2] = (u - v + MOD) % MOD
                    w = (w * wlen) % MOD
            len_ntt *= 2

        if invert:
            n_inv = pow(n, MOD - 2, MOD)
            for i in range(n):
                a[i] = (a[i] * n_inv) % MOD
        return a

    # Pad to next power of 2 for NTT length
    len_dp_poly = N + 1 # dp[0]...dp[N]
    len_B_poly = N      # B[0]...B[N-1]
    
    NTT_LEN = 1
    while NTT_LEN < len_dp_poly + len_B_poly - 1:
        NTT_LEN *= 2
    
    A_poly = list(dp) + [0] * (NTT_LEN - len_dp_poly)
    B_poly = list(B) + [0] * (NTT_LEN - len_B_poly)

    A_poly_ntt = ntt(A_poly, False)
    B_poly_ntt = ntt(B_poly, False)

    C_ntt = [ (A_poly_ntt[i] * B_poly_ntt[i]) % MOD for i in range(NTT_LEN) ]
    C = ntt(C_ntt, True) # C contains coefficients C[0]...C[2N-1]

    # 6. Compute Total Sum = Sum_{q=0}^{N-1} (N-q-1)! * q! * C_q
    total_sum = 0
    for q in range(N): # q goes from 0 to N-1
        # C_q is the coefficient of z^q from (dp * B)
        # (N-q-1)! is fact[N-q-1]
        # q! is fact[q]
        term_q_product = (fact[N-q-1] * fact[q]) % MOD
        total_sum = (total_sum + term_q_product * C[q]) % MOD

    sys.stdout.write(str(total_sum) + '
')

solve()