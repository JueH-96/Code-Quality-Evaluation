class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        MOD = 1000000007
        binom_n = n - 1
        binom_k = n - k - 1  # s - 1 where s is the number of runs
        
        # Compute factorial up to binom_n
        fact = [1] * (binom_n + 1)
        for i in range(1, binom_n + 1):
            fact[i] = (fact[i - 1] * i) % MOD
        
        # Compute modular inverses
        inv_fact_bk = pow(fact[binom_k], MOD - 2, MOD)
        inv_fact_diff = pow(fact[k], MOD - 2, MOD)  # inv of fact[k], since binom_n - binom_k = k
        
        # Binomial coefficient C(binom_n, binom_k)
        binom_coeff = (fact[binom_n] * inv_fact_bk % MOD * inv_fact_diff % MOD) % MOD
        
        # Compute the power part (m-1)^binom_k mod MOD
        pow_part = pow(m - 1, binom_k, MOD)
        
        # m part
        m_part = m  # m < MOD, no need for modulo
        
        # Final result
        result = (binom_coeff * m_part % MOD * pow_part % MOD) % MOD
        return result