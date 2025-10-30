class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        MOD = 10**9 + 7
        s = n - k
        if s < 1:
            return 0
        
        max_fact = n - 1
        fact = [1] * (max_fact + 1)
        for i in range(1, max_fact + 1):
            fact[i] = fact[i-1] * i % MOD
        
        inv_fact = [1] * (max_fact + 1)
        if max_fact >= 0:
            inv_fact[max_fact] = pow(fact[max_fact], MOD - 2, MOD)
            for i in range(max_fact - 1, -1, -1):
                inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD
        
        a = n - 1
        b = s - 1
        if b < 0 or b > a:
            return 0
        c = fact[a] * inv_fact[b] % MOD
        c = c * inv_fact[a - b] % MOD
        
        pow_val = pow(m - 1, s - 1, MOD)
        ans = c * m % MOD
        ans = ans * pow_val % MOD
        return ans