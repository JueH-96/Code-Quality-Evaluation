class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        target_set = set(targetIndices)
        n, m = len(source), len(pattern)
        
        # dp[i][j] = minimum number of characters from target_set needed 
        # to match pattern[0:i] using source[0:j]
        dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]
        
        # Base case: no pattern to match requires 0 characters
        for j in range(n + 1):
            dp[0][j] = 0
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Option 1: Don't use source[j-1]
                dp[i][j] = dp[i][j-1]
                
                # Option 2: Use source[j-1] to match pattern[i-1]
                if source[j-1] == pattern[i-1]:
                    cost = dp[i-1][j-1] + (1 if j-1 in target_set else 0)
                    dp[i][j] = min(dp[i][j], cost)
        
        # Return max removals = total target indices - minimum we must keep
        return len(targetIndices) - dp[m][n]