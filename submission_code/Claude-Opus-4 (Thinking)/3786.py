class Solution:
    def longestPalindromicSubsequence(self, s: str, k: int) -> int:
        n = len(s)
        
        def min_ops_to_equal(c1, c2):
            if c1 == c2:
                return 0
            a = ord(c1) - ord('a')
            b = ord(c2) - ord('a')
            diff = abs(a - b)
            return min(diff, 26 - diff)
        
        memo = {}
        
        def solve(i, j, ops):
            if i > j:
                return 0
            if i == j:
                return 1
            if (i, j, ops) in memo:
                return memo[(i, j, ops)]
            
            result = 0
            
            # Try to match s[i] and s[j]
            cost = min_ops_to_equal(s[i], s[j])
            if cost <= ops:
                result = max(result, 2 + solve(i + 1, j - 1, ops - cost))
            
            # Skip s[i]
            result = max(result, solve(i + 1, j, ops))
            
            # Skip s[j]
            result = max(result, solve(i, j - 1, ops))
            
            memo[(i, j, ops)] = result
            return result
        
        return solve(0, n - 1, k)