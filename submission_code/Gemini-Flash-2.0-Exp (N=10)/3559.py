from typing import List

class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0

        for i in range(1, n + 1):
            for word in words:
                for j in range(1, len(word) + 1):
                    prefix = word[:j]
                    if i >= len(prefix) and target[i - len(prefix):i] == prefix:
                        dp[i] = min(dp[i], dp[i - len(prefix)] + 1)
        
        return dp[n] if dp[n] != float('inf') else -1