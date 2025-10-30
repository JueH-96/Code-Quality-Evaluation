class Solution:
    def longestPalindromicSubsequence(self, s: str, k: int) -> int:
        n = len(s)
        
        # Function to calculate minimum cost to make two characters equal
        def min_cost(c1, c2):
            if c1 == c2:
                return 0
            
            # Convert to numbers (0-25)
            a = ord(c1) - ord('a')
            b = ord(c2) - ord('a')
            
            # Calculate circular distance
            forward = (b - a) % 26
            backward = (a - b) % 26
            
            return min(forward, backward)
        
        # DP with memoization
        # dp[i][j][cost] = longest palindromic subsequence in s[i:j+1] with exactly 'cost' operations used
        from functools import lru_cache
        
        @lru_cache(None)
        def dp(i, j, cost):
            if cost > k:
                return -float('inf')
            
            if i > j:
                return 0
            
            if i == j:
                return 1 if cost == 0 else -float('inf')
            
            # Option 1: Skip both characters
            result = dp(i + 1, j - 1, cost)
            
            # Option 2: Skip left character
            result = max(result, dp(i + 1, j, cost))
            
            # Option 3: Skip right character
            result = max(result, dp(i, j - 1, cost))
            
            # Option 4: Include both characters if we can make them equal
            change_cost = min_cost(s[i], s[j])
            if cost + change_cost <= k:
                result = max(result, 2 + dp(i + 1, j - 1, cost + change_cost))
            
            return result
        
        # Try all possible costs from 0 to k
        max_length = 0
        for cost in range(k + 1):
            max_length = max(max_length, dp(0, n - 1, cost))
        
        return max_length