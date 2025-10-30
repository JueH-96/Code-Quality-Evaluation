MOD = 10**9 + 7

class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        a = k + (n - 1)
        b = n - 1
        if a < b:
            return 0
        
        # Compute factorial up to 'a'
        max_fact = a
        fact = [1] * (max_fact + 1)
        for i in range(1, max_fact + 1):
            fact[i] = fact[i-1] * i % MOD
        
        # Compute inverse factorial using Fermat's Little Theorem
        inv_fact = [1] * (max_fact + 1)
        inv_fact[max_fact] = pow(fact[max_fact], MOD - 2, MOD)
        for i in range(max_fact - 1, -1, -1):
            inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD
        
        # Calculate the binomial coefficient C(a, b)
        if b < 0 or b > a:
            return 0
        res = fact[a] * inv_fact[b] % MOD
        res = res * inv_fact[a - b] % MOD
        return res