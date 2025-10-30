import math

class Solution:
    def distanceSum(self, m: int, n: int, k: int) -> int:
        MOD = 1000000007
        N = m * n
        a = N - 2
        b = k - 2
        
        # Compute binomial coefficient C(a, b) mod MOD
        fact = [1] * (a + 1)
        for i in range(1, a + 1):
            fact[i] = (fact[i - 1] * i) % MOD
        inv_fb = pow(fact[b], MOD - 2, MOD)
        inv_fa_b = pow(fact[a - b], MOD - 2, MOD)
        binom_val = (fact[a] * inv_fb % MOD * inv_fa_b % MOD) % MOD
        
        # Compute dist_sum_part
        inv6 = pow(6, MOD - 2, MOD)
        prod = (N * (m + n) % MOD) % MOD
        prod = (prod * ((N - 1) % MOD) % MOD) % MOD
        dist_sum_val = (prod * inv6 % MOD) % MOD
        
        # Compute result
        result = (dist_sum_val * binom_val % MOD) % MOD
        return result