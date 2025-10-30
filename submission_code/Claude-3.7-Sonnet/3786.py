class Solution:
    def longestPalindromicSubsequence(self, s: str, k: int) -> int:
        n = len(s)
        
        # Calculate the cost to make two characters match
        def cost_to_match(a, b):
            if a == b:
                return 0
            return min(abs(ord(a) - ord(b)), 26 - abs(ord(a) - ord(b)))
        
        # Memoization
        memo = {}
        
        def dp(i, j, ops):
            if (i, j, ops) in memo:
                return memo[(i, j, ops)]
            
            if i > j:
                return 0
            if i == j:
                return 1
            
            # Option 1: Skip s[i]
            skip_i = dp(i+1, j, ops)
            
            # Option 2: Skip s[j]
            skip_j = dp(i, j-1, ops)
            
            # Option 3: Include both s[i] and s[j]
            include_both = 0
            if s[i] == s[j]:
                include_both = 2 + dp(i+1, j-1, ops)
            else:
                cost = cost_to_match(s[i], s[j])
                if ops >= cost:
                    include_both = 2 + dp(i+1, j-1, ops - cost)
            
            result = max(skip_i, skip_j, include_both)
            memo[(i, j, ops)] = result
            return result
        
        return dp(0, n-1, k)