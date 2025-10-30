class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        n = len(target)
        # dp[i] represents min number of valid strings to form target[0:i]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(n):
            if dp[i] == float('inf'):
                continue
                
            # Try each word starting from position i
            for word in words:
                # Find the longest matching prefix
                j = 0
                while j < len(word) and i + j < n and word[j] == target[i + j]:
                    j += 1
                
                # Update dp for all valid prefix lengths
                for k in range(1, j + 1):
                    dp[i + k] = min(dp[i + k], dp[i] + 1)
        
        return dp[n] if dp[n] != float('inf') else -1