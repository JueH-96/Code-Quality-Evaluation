from typing import List
import sys

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        n = len(target)
        dp = [sys.maxsize] * (n + 1)
        dp[0] = 0
        
        for i in range(1, n + 1):
            for j in range(len(words)):
                word = words[j]
                cost = costs[j]
                if i >= len(word) and target[i - len(word):i] == word:
                    dp[i] = min(dp[i], dp[i - len(word)] + cost)
        
        return dp[n] if dp[n] != sys.maxsize else -1