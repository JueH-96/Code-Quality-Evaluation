from typing import List
import collections
import bisect

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        word_cost = collections.defaultdict(lambda: float('inf'))
        
        # Store the minimum cost for each word
        for word, cost in zip(words, costs):
            word_cost[word] = min(word_cost[word], cost)
        
        # Sort words by length for efficient prefix matching
        words = sorted(word_cost.keys(), key=len)
        
        for i in range(1, n + 1):
            for word in words:
                if target[i - len(word):i] == word:
                    dp[i] = min(dp[i], dp[i - len(word)] + word_cost[word])
        
        return dp[n] if dp[n] != float('inf') else -1