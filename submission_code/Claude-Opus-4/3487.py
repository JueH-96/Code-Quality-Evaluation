class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        n = len(source)
        m = len(pattern)
        k = len(targetIndices)
        
        # Convert targetIndices to a set for O(1) lookup
        target_set = set(targetIndices)
        
        # dp[i][j] = minimum number of characters from targetIndices we must keep
        # to match pattern[0:j] using source[0:i]
        # Initialize with infinity
        dp = [[float('inf')] * (m + 1) for _ in range(n + 1)]
        
        # Base case: empty pattern requires 0 characters to keep
        for i in range(n + 1):
            dp[i][0] = 0
        
        # Fill the dp table
        for i in range(1, n + 1):
            for j in range(m + 1):
                # Option 1: Skip source[i-1]
                if i - 1 in target_set:
                    # If we skip a character in targetIndices, we don't need to keep it
                    dp[i][j] = dp[i-1][j]
                else:
                    # If it's not in targetIndices, just inherit the value
                    dp[i][j] = dp[i-1][j]
                
                # Option 2: Use source[i-1] to match pattern[j-1]
                if j > 0 and source[i-1] == pattern[j-1]:
                    if i - 1 in target_set:
                        # If we use a character from targetIndices, we must keep it
                        dp[i][j] = min(dp[i][j], dp[i-1][j-1] + 1)
                    else:
                        # If it's not in targetIndices, no cost
                        dp[i][j] = min(dp[i][j], dp[i-1][j-1])
        
        # The answer is total targetIndices minus minimum we must keep
        min_keep = dp[n][m]
        if min_keep == float('inf'):
            return 0
        
        return k - min_keep