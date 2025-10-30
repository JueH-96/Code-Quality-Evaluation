from typing import List
import collections

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        word_cost = {word: cost for word, cost in zip(words, costs)}
        
        for i in range(n + 1):
            if dp[i] == float('inf'):
                continue
            for word in word_cost:
                if target[i:i + len(word)] == word:
                    dp[i + len(word)] = min(dp[i + len(word)], dp[i] + word_cost[word])
        
        return dp[n] if dp[n] != float('inf') else -1