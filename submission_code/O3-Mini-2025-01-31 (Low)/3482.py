from typing import List
from collections import defaultdict
import math

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        n = len(target)
        # Group words by their first letter along with their cost.
        groups = defaultdict(list)
        for w, c in zip(words, costs):
            groups[w[0]].append((w, c))
        
        # dp[i] = minimum cost to build target[:i]
        dp = [math.inf] * (n + 1)
        dp[0] = 0
        
        for i in range(n):
            if dp[i] == math.inf:
                continue
            # get all words starting with target[i]
            first_letter = target[i]
            if first_letter not in groups:
                continue
            for word, cost in groups[first_letter]:
                lenw = len(word)
                end = i + lenw
                if end > n:  # can't fit in target substring
                    continue
                # Check if word exactly matches target substring starting at i.
                if target[i:end] == word:
                    dp[end] = min(dp[end], dp[i] + cost)
                    
        return dp[n] if dp[n] != math.inf else -1