class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        MOD = 10**9 + 7
        a = n + k - 1
        b = n - 1
        
        if a == 0 or b == 0:
            return 1
        
        max_fact = a
        fact = [1] * (max_fact + 1)
        for i in range(1, max_fact + 1):
            fact[i] = fact[i-1] * i % MOD
        
        inv_fact = [1] * (max_fact + 1)
        inv_fact[max_fact] = pow(fact[max_fact], MOD - 2, MOD)
        for i in range(max_fact - 1, -1, -1):
            inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD
        
        return fact[a] * inv_fact[b] % MOD * inv_fact[a - b] % MOD