class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        n, m = len(source), len(pattern)
        target_set = set(targetIndices)
        
        # dp[i][j] = minimum number of characters from targetIndices 
        # that we must keep to match pattern[0:j] using source[0:i]
        dp = [[float('inf')] * (m + 1) for _ in range(n + 1)]
        
        # Base case: empty pattern requires keeping 0 characters from targetIndices
        for i in range(n + 1):
            dp[i][0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # Option 1: skip source[i-1]
                dp[i][j] = dp[i-1][j]
                
                # Option 2: use source[i-1] to match pattern[j-1] (if they match)
                if source[i-1] == pattern[j-1]:
                    cost = 1 if (i-1) in target_set else 0
                    dp[i][j] = min(dp[i][j], dp[i-1][j-1] + cost)
        
        # Maximum removals = total in targetIndices - minimum we must keep
        return len(targetIndices) - dp[n][m]