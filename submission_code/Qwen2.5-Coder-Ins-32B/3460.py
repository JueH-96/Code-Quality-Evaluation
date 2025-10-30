from typing import List
from functools import lru_cache

MOD = 10**9 + 7

class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        # Convert requirements to a dictionary for easier access
        req_dict = {end: cnt for end, cnt in requirements}
        
        # Precompute factorials and their modular inverses for combinatorial calculations
        fact = [1] * (n + 1)
        inv_fact = [1] * (n + 1)
        for i in range(2, n + 1):
            fact[i] = fact[i - 1] * i % MOD
            inv_fact[i] = pow(fact[i], MOD - 2, MOD)
        
        # Function to compute nCr % MOD
        def comb(n, r):
            if r > n:
                return 0
            return fact[n] * inv_fact[r] % MOD * inv_fact[n - r] % MOD
        
        # DP function to count permutations
        @lru_cache(None)
        def dp(i, inv):
            if i == -1:
                return 1 if inv == 0 else 0
            end = i
            min_inv = req_dict.get(end, 0)
            max_inv = inv
            count = 0
            for j in range(min_inv, max_inv + 1):
                count = (count + comb(i, j) * dp(i - 1, inv - j)) % MOD
            return count
        
        return dp(n - 1, req_dict[n - 1])