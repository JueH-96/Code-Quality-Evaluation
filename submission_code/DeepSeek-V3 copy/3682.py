MOD = 10**9 + 7

class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        if k >= n:
            return 0
        
        # Precompute factorials and inverse factorials modulo MOD
        max_n = n
        fact = [1] * (max_n + 1)
        inv_fact = [1] * (max_n + 1)
        
        for i in range(1, max_n + 1):
            fact[i] = fact[i-1] * i % MOD
        
        inv_fact[max_n] = pow(fact[max_n], MOD-2, MOD)
        for i in range(max_n - 1, -1, -1):
            inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
        
        def comb(n, k):
            if k < 0 or k > n:
                return 0
            return fact[n] * inv_fact[k] % MOD * inv_fact[n - k] % MOD
        
        # The number of ways to choose k positions out of n-1 to have arr[i-1] == arr[i]
        ways = comb(n-1, k)
        
        # The number of ways to assign values to the array such that the chosen positions have equal values
        # The first element can be any of the m values
        # For each of the k chosen positions, the value must be the same as the previous
        # For the remaining positions, the value must be different from the previous
        # So, the total is m * (1)^k * (m-1)^(n-1 -k)
        total = m * pow(1, k, MOD) * pow(m-1, n-1 -k, MOD) % MOD
        
        # Multiply the number of ways to choose positions with the number of ways to assign values
        result = ways * total % MOD
        
        return result