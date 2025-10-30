from typing import List

class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(1, n + 1):
            for word in words:
                if i >= len(word) and target[i-len(word):i] == word:
                    dp[i] = min(dp[i], dp[i-len(word)] + 1)
                elif target[:i].startswith(word):
                    dp[i] = min(dp[i], dp[i-len(word)] + 1)
        
        return dp[n] if dp[n] != float('inf') else -1