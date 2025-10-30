class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        MOD = 10**9 + 7
        if m == 1:
            if k == n - 1:
                return 1
            else:
                return 0
        
        def power(a, b):
            res = 1
            a %= MOD
            while b > 0:
                if b % 2 == 1:
                    res = (res * a) % MOD
                a = (a * a) % MOD
                b //= 2
            return res
            
        def nCr_mod_p(n_val, r_val, fact, inv_fact):
            if r_val < 0 or r_val > n_val:
                return 0
            num = fact[n_val]
            den = (inv_fact[r_val] * inv_fact[n_val - r_val]) % MOD
            return (num * den) % MOD
            
        max_val = n
        fact = [1] * (max_val + 1)
        inv_fact = [1] * (max_val + 1)
        for i in range(1, max_val + 1):
            fact[i] = (fact[i - 1] * i) % MOD
        inv_fact[max_val] = power(fact[max_val], MOD - 2)
        for i in range(max_val - 1, -1, -1):
            inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD
            
        nCr_val = nCr_mod_p(n - 1, k, fact, inv_fact)
        power_val = power(m - 1, n - k - 1)
        
        result = (nCr_val * m) % MOD
        result = (result * power_val) % MOD
        return result