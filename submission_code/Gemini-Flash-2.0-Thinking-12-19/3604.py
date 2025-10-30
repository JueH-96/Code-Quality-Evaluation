class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 10**9 + 7
        
        def nCr_mod_p(n, r, fact, inv_fact, p):
            if r < 0 or r > n:
                return 0
            return (fact[n] * inv_fact[r] * inv_fact[n - r]) % p
            
        def power(base, exp, p):
            res = 1
            base %= p
            while exp > 0:
                if exp % 2 == 1:
                    res = (res * base) % p
                exp >>= 1
                base = (base * base) % p
            return res
            
        max_val = max(x, n)
        fact = [1] * (max_val + 1)
        inv_fact = [1] * (max_val + 1)
        for i in range(1, max_val + 1):
            fact[i] = (fact[i - 1] * i) % MOD
        inv_fact[max_val] = power(fact[max_val], MOD - 2, MOD)
        for i in range(max_val - 1, -1, -1):
            inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD
            
        total_ways = 0
        min_kx = min(n, x)
        for k in range(1, min_kx + 1):
            s_nk = 0
            for j in range(k + 1):
                term = (nCr_mod_p(k, j, fact, inv_fact, MOD) * power(k - j, n, MOD)) % MOD
                if j % 2 == 0:
                    s_nk = (s_nk + term) % MOD
                else:
                    s_nk = (s_nk - term + MOD) % MOD
            term_k = (nCr_mod_p(x, k, fact, inv_fact, MOD) * s_nk) % MOD
            term_k = (term_k * power(y, k, MOD)) % MOD
            total_ways = (total_ways + term_k) % MOD
            
        return total_ways