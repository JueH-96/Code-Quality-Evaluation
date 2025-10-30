from collections import defaultdict
from typing import List

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        # Group words by their first character to reduce the number of comparisons
        candidates = defaultdict(list)
        for word, cost in zip(words, costs):
            candidates[word[0]].append((word, cost))
        
        n = len(target)
        # dp[i] = minimum cost to form target[:i]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # empty string cost is 0
        
        for i in range(n):
            if dp[i] == float('inf'):
                continue
            # Check only words that can start with target[i]
            for word, cost in candidates[target[i]]:
                wlen = len(word)
                if i + wlen <= n and target[i:i+wlen] == word:
                    dp[i+wlen] = min(dp[i+wlen], dp[i] + cost)
                    
        return dp[n] if dp[n] != float('inf') else -1