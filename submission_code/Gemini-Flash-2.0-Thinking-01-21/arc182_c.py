def solve():
    n, m = map(int, input().split())
    if m == 1:
        print(n % 998244353)
        return
    
    primes_le_m = []
    primes = [2, 3, 5, 7, 11, 13, 17, 19]
    for p in primes:
        if p <= m:
            primes_le_m.append(p)
        else:
            break
    
    v_values = []
    for p in primes_le_m:
        v_p = 0
        power = p
        while power <= m:
            v_p += m // power
            if power > m // p:
                break
            power *= p
        v_values.append(v_p)
        
    num_primes = len(primes_le_m)
    
    def power(a, b, mod):
        res = 1
        a %= mod
        while b > 0:
            if b % 2 == 1:
                res = (res * a) % mod
            a = (a * a) % mod
            b //= 2
        return res
        
    def inverse(n, mod):
        return power(n, mod - 2, mod)
        
    mod_val = 998244353
    
    def get_T_r(r, m, n, mod):
        if m == 1:
            return n if r == 0 else 0
        
        def f_0(x, n, mod):
            num = (power(x, n + 1, mod) - x + mod) % mod
            den_inv = inverse(x - 1 + mod, mod)
            return (num * den_inv) % mod
            
        f_r_values = {}
        f_r_values[0] = f_0(m, n, mod)
        
        def derivative_f_r(r, x, n, mod):
            if r == 0:
                num = (n * power(x, n + 1, mod) - (n + 1) * power(x, n, mod) + 1 + mod) % mod
                den_sq_inv = inverse(power(x - 1 + mod, 2, mod), mod)
                return (num * den_sq_inv) % mod
            else:
                # We don't need to calculate derivative explicitly for r > 0, 
                # as we only need f_r(m) values.
                # Let's calculate f_r(m) recursively.
                return 0 # Placeholder, not used in actual calculation.

        def f_next(r, x, n, mod):
            return (x * derivative_f_r(r, x, n, mod)) % mod

        for current_r in range(r):
            if current_r + 1 <= r:
                f_r_values[current_r + 1] = (m * derivative_f_r(current_r, m, n, mod)) % mod
                # Actually, we need to use recursive definition f_{r+1}(x) = x * f'_r(x).
                # But we can simplify for our purpose. 
                # We need T_r = M^{-r} f_r(M). So, T_r = sum_{k=1}^N k^r M^{k-r}.
                # T_0 = sum_{k=1}^N M^k = (M^(N+1) - M) / (M-1).
                # T_1 = sum_{k=1}^N k M^{k-1}. T_2 = sum_{k=1}^N k^2 M^{k-2}. 
                # Instead of recursion, calculate T_r directly.
                pass

        t_r_val = 0
        for k in range(1, n + 1):
            term = (power(k, r, mod) * power(m, k - r, mod)) % mod
            t_r_val = (t_r_val + term) % mod
        return t_r_val

    total_score_sum = 0
    
    for i in range(1 << num_primes):
        product_v = 1
        count_set_bits = 0
        indices_in_set = []
        for j in range(num_primes):
            if (i >> j) & 1:
                product_v = (product_v * v_values[j]) % mod_val
                count_set_bits += 1
                
        t_r = get_T_r(count_set_bits, m, n, mod_val)
        term_val = (product_v * t_r) % mod_val
        total_score_sum = (total_score_sum + term_val) % mod_val
        
    print(total_score_sum)

if __name__ == '__main__':
    solve()