class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        # For each position in target
        for i in range(n):
            if dp[i] == float('inf'):
                continue
                
            # Try each word
            for word in words:
                # Get maximum length we can match starting at position i
                matched_len = 0
                for j in range(len(word)):
                    if i + j >= n or target[i + j] != word[j]:
                        break
                    matched_len = j + 1
                
                # If we found a valid prefix
                if matched_len > 0:
                    dp[i + matched_len] = min(dp[i + matched_len], dp[i] + 1)
        
        return dp[n] if dp[n] != float('inf') else -1