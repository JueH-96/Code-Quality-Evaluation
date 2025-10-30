from typing import List
import sys

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        n = len(target)
        dp = [sys.maxsize] * (n + 1)
        dp[0] = 0
        
        for i in range(n):
            if dp[i] == sys.maxsize:
                continue
            for j, word in enumerate(words):
                if target.startswith(word, i):
                    dp[i + len(word)] = min(dp[i + len(word)], dp[i] + costs[j])
        
        return dp[n] if dp[n] != sys.maxsize else -1