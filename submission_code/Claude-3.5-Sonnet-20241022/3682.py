class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        MOD = 10**9 + 7
        
        # dp[i][j] represents number of arrays of length i with j equal adjacent pairs
        dp = {}
        
        def solve(pos, equal_pairs, prev):
            if equal_pairs > k:  # If we exceed required equal pairs
                return 0
            
            if pos == n:  # If we reach the end
                return 1 if equal_pairs == k else 0
            
            state = (pos, equal_pairs, prev)
            if state in dp:
                return dp[state]
            
            result = 0
            # Try all possible numbers from 1 to m
            for num in range(1, m + 1):
                # If previous number exists, check if it forms an equal pair
                new_pairs = equal_pairs
                if pos > 0 and num == prev:
                    new_pairs += 1
                result = (result + solve(pos + 1, new_pairs, num)) % MOD
            
            dp[state] = result
            return result
        
        # Start with position 0, 0 equal pairs, and prev as None
        total = 0
        for first in range(1, m + 1):
            total = (total + solve(1, 0, first)) % MOD
            
        return total