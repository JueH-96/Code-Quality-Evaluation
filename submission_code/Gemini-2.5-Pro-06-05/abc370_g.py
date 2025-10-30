import sys

def solve():
    """
    Main function to solve the problem.
    """
    N, M = map(int, sys.stdin.readline().split())
    MOD = 998244353

    # Define arithmetic in F_p(omega) where omega^2 + omega + 1 = 0
    def add(c1, c2):
        return (c1[0] + c2[0]) % MOD, (c1[1] + c2[1]) % MOD

    def sub(c1, c2):
        return (c1[0] - c2[0] + MOD) % MOD, (c1[1] - c2[1] + MOD) % MOD

    def mul(c1, c2):
        a, b = c1
        c, d = c2
        ac = (a * c) % MOD
        bd = (b * d) % MOD
        ad_plus_bc = (a * d + b * c) % MOD
        real = (ac - bd + MOD) % MOD
        imag = (ad_plus_bc - bd + MOD) % MOD
        return real, imag

    # Precompute combinations
    # Max exponent for a prime is log2(N) < 40. 60 is a safe upper bound.
    MAX_E = 60
    max_comb_n = M + MAX_E
    fact = [1] * (max_comb_n + 1)
    inv_fact = [1] * (max_comb_n + 1)
    for i in range(1, max_comb_n + 1):
        fact[i] = (fact[i - 1] * i) % MOD
    inv_fact[max_comb_n] = pow(fact[max_comb_n], MOD - 2, MOD)
    for i in range(max_comb_n - 1, -1, -1):
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD

    memo_comb = {}
    def combinations(n, k):
        if k < 0 or k > n: return 0
        state = (n, k)
        if state in memo_comb: return memo_comb[state]
        if k == 0 or k == n: return 1
        if k > n // 2: k = n - k
        
        if n <= max_comb_n:
            res = (fact[n] * inv_fact[k] * inv_fact[n-k]) % MOD
        else: # For large n, k is small (M-1)
            res = inv_fact[k]
            for i in range(k):
                res = res * (n - i) % MOD
        memo_comb[state] = res
        return res

    memo_f_pe = [{}, {}, {}]
    def get_f_pe_func(j):
        omega = (0, 1)
        w2 = (-1, -1)
        
        def f_pe(p, e):
            state = (p, e)
            if state in memo_f_pe[j]: return memo_f_pe[j][state]

            comb = combinations(e + M - 1, e)
            
            if j == 0:
                res = (comb, 0)
                memo_f_pe[j][state] = res
                return res

            if p % 3 == 0: g_val = 1
            elif p % 3 == 1: g_val = (e + 1) % 3
            else: g_val = 1 if e % 2 == 0 else 0
            
            power = (-j * g_val % 3 + 3) % 3
            
            w_pow = (1, 0)
            if power == 1: w_pow = omega
            elif power == 2: w_pow = w2
            
            res = (comb * w_pow[0] % MOD, comb * w_pow[1] % MOD)
            memo_f_pe[j][state] = res
            return res
        return f_pe

    def sum_multiplicative(N_val, f_pe):
        if N_val == 0: return (0, 0)
        
        SQRT_N = int(N_val**0.5)
        primes = []
        min_prime_factor = list(range(SQRT_N + 1))
        for i in range(2, SQRT_N + 1):
            if min_prime_factor[i] == i:
                primes.append(i)
            for p in primes:
                if p > min_prime_factor[i] or i * p > SQRT_N: break
                min_prime_factor[i * p] = p
        
        pi_SQRT_N = len(primes)
        
        vals = []
        i = 1
        while i <= N_val:
            vals.append(i)
            i = N_val // (N_val // i) + 1
        
        val_to_idx = {v: i for i, v in enumerate(vals)}
        
        # S(n,k) = sum f(i) for i<=n with spf(i) > p_k
        S = [(v, 0) for v in vals] 

        for k in range(pi_SQRT_N - 1, -1, -1):
            p = primes[k]
            pp = p * p
            for i in range(len(vals) - 1, -1, -1):
                v = vals[i]
                if v < pp: break
                pe = p
                for e in range(1, MAX_E):
                    if pe > v: break
                    f_pe_val = f_pe(p, e)
                    
                    v_div_pe = v // pe
                    idx = val_to_idx[v_div_pe]
                    term = add(S[idx], (1,0))
                    S[i] = add(S[i], mul(f_pe_val, term))

                    if pe > v // p: break
                    pe *= p
        
        return add(S[val_to_idx[N_val]], (1,0))

    f_pe_0 = get_f_pe_func(0)
    s0 = sum_multiplicative(N, f_pe_0)
    
    f_pe_1 = get_f_pe_func(1)
    s1 = sum_multiplicative(N, f_pe_1)
    
    a0 = s0[0]
    a1, b1 = s1
    
    # S2 is conjugate of S1, so S2 = (a1-b1, -b1)
    # Sum = S0 + S1 + S2 = (a0,0) + (a1,b1) + (a1-b1, -b1) = (a0 + 2*a1 - b1, 0)
    res = (a0 + 2 * a1 - b1 + MOD) % MOD
    inv3 = pow(3, MOD - 2, MOD)
    res = (res * inv3) % MOD
    
    print(res)

solve()