from typing import List
from collections import defaultdict

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        word_index = defaultdict(list)
        for i, word in enumerate(words):
            word_index[word].append(i)
        
        for i in range(1, n + 1):
            for j in range(i):
                for idx in word_index[target[j:i]]:
                    dp[i] = min(dp[i], dp[j] + costs[idx])
        
        return dp[n] if dp[n] != float('inf') else -1