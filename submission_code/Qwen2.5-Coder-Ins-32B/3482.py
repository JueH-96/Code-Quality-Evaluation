from typing import List
import functools

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        @functools.lru_cache(None)
        def dp(i):
            if i == len(target):
                return 0
            if i > len(target):
                return float('inf')
            
            min_cost = float('inf')
            for word, cost in zip(words, costs):
                if target.startswith(word, i):
                    min_cost = min(min_cost, cost + dp(i + len(word)))
            
            return min_cost
        
        result = dp(0)
        return result if result != float('inf') else -1