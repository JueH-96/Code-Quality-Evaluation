class Solution:
    MOD = 10**9 + 7

    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        t = (n - 1) - k
        if t < 0 or t > n - 1:
            return 0
        
        if n == 0:
            return 0
        
        max_needed = max(n - 1, t)
        fact = [1] * (max_needed + 1)
        for i in range(1, max_needed + 1):
            fact[i] = fact[i-1] * i % Solution.MOD
        
        inv_fact = [1] * (max_needed + 1)
        inv_fact[max_needed] = pow(fact[max_needed], Solution.MOD - 2, Solution.MOD)
        for i in range(max_needed - 1, -1, -1):
            inv_fact[i] = inv_fact[i + 1] * (i + 1) % Solution.MOD
        
        if t == 0:
            comb = 1
        else:
            comb = fact[n - 1] * inv_fact[t] % Solution.MOD
            comb = comb * inv_fact[n - 1 - t] % Solution.MOD
        
        if m == 1:
            if t == 0:
                pow_term = 1
            else:
                pow_term = 0
        else:
            pow_term = pow(m - 1, t, Solution.MOD)
        
        total = comb * m % Solution.MOD
        total = total * pow_term % Solution.MOD
        return total