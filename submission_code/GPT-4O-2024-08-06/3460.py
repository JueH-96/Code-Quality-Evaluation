from typing import List
from math import factorial

MOD = 10**9 + 7

class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        # Precompute factorials and inverse factorials for n
        fact = [1] * (n + 1)
        inv_fact = [1] * (n + 1)
        
        for i in range(2, n + 1):
            fact[i] = fact[i - 1] * i % MOD
        
        inv_fact[n] = pow(fact[n], MOD - 2, MOD)
        for i in range(n - 1, 0, -1):
            inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD
        
        # Function to calculate nCk % MOD
        def comb(n, k):
            if k > n or k < 0:
                return 0
            return fact[n] * inv_fact[k] % MOD * inv_fact[n - k] % MOD
        
        # dp[i][j] = number of permutations of [0..i] with exactly j inversions
        dp = [[0] * (n * (n - 1) // 2 + 1) for _ in range(n)]
        
        # Base case: 0 inversions with 1 element
        dp[0][0] = 1
        
        # Fill the dp table
        for i in range(1, n):
            for j in range(n * (n - 1) // 2 + 1):
                for k in range(min(j, i) + 1):
                    dp[i][j] = (dp[i][j] + dp[i - 1][j - k] * comb(i, k)) % MOD
        
        # Check all requirements
        result = 0
        for perm in range(fact[n]):
            valid = True
            for end, cnt in requirements:
                if dp[end][cnt] == 0:
                    valid = False
                    break
            if valid:
                result = (result + 1) % MOD
        
        return result