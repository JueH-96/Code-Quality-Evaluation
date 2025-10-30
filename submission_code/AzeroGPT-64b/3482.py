from collections import defaultdict
from functools import lru_cache

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        
        wordMap = defaultdict(list)
        for idx, word in enumerate(words):
            wordMap[word].append((idx, costs[idx]))
        
        @lru_cache(None)
        def dp(i, s):
            if i >= len(target): return 0 if len(s) == len(target) else float("inf")
            if len(s) > len(target): return float("inf")
            res = float("inf")
            if s and target[i] == s[-1]:
                res = dp(i+1, s[:-1])
            res = min(res, dp(i, s) + dp(i+1, s))
            for word, cost in wordMap[target[i]]:
                if not s or word[0] == s[-1]:
                    res = min(res, cost + dp(i, s + word[1:]))
            return res
        
        res = dp(0, "")
        return res if res != float("inf") else -1