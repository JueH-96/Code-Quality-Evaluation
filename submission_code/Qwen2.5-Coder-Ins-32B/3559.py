from typing import List

class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        prefixes = set()
        for word in words:
            for i in range(1, len(word) + 1):
                prefixes.add(word[:i])
        
        dp = [float('inf')] * (len(target) + 1)
        dp[0] = 0
        
        for i in range(1, len(target) + 1):
            for prefix in prefixes:
                if i >= len(prefix) and target[i - len(prefix):i] == prefix:
                    dp[i] = min(dp[i], dp[i - len(prefix)] + 1)
        
        return dp[-1] if dp[-1] != float('inf') else -1