class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        n = len(target)
        # dp[i] represents minimum number of valid strings to form target[0:i]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Empty string needs 0 valid strings
        
        # For each position in target
        for i in range(n):
            if dp[i] == float('inf'):
                continue
                
            # Try each word
            for word in words:
                # Check how long of a prefix of word matches target starting at position i
                j = 0
                while j < len(word) and i + j < n and word[j] == target[i + j]:
                    j += 1
                    # Update dp for position i + j
                    dp[i + j] = min(dp[i + j], dp[i] + 1)
        
        return dp[n] if dp[n] != float('inf') else -1