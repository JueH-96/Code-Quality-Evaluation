class Solution:
    def numberOfWays(self, n: int, x: int, y: int) -> int:
        MOD = 10**9 + 7
        
        def power(a, b):
            res = 1
            a %= MOD
            while b > 0:
                if b % 2 == 1:
                    res = (res * a) % MOD
                a = (a * a) % MOD
                b //= 2
            return res
            
        def inverse(n):
            return power(n, MOD - 2)
            
        MAX_VAL = 1001
        fact = [1] * MAX_VAL
        inv_fact = [1] * MAX_VAL
        for i in range(1, MAX_VAL):
            fact[i] = (fact[i-1] * i) % MOD
        inv_fact[MAX_VAL-1] = inverse(fact[MAX_VAL-1])
        for i in range(MAX_VAL-2, -1, -1):
            inv_fact[i] = (inv_fact[i+1] * (i + 1)) % MOD
            
        def nCr_mod(n, r):
            if r < 0 or r > n:
                return 0
            num = fact[n]
            den = (inv_fact[r] * inv_fact[n-r]) % MOD
            return (num * den) % MOD
            
        total_ways = 0
        for k in range(1, x + 1):
            inner_sum = 0
            for i in range(k + 1):
                term = (nCr_mod(k, i) * power(k - i, n)) % MOD
                if i % 2 == 1:
                    inner_sum = (inner_sum - term + MOD) % MOD
                else:
                    inner_sum = (inner_sum + term) % MOD
            term_k = (nCr_mod(x, k) * inner_sum) % MOD
            term_k = (term_k * power(y, k)) % MOD
            total_ways = (total_ways + term_k) % MOD
            
        return total_ways