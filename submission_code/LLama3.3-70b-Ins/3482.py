from typing import List

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        memo = {}

        def dp(i: int) -> int:
            if i == len(target):
                return 0
            if i in memo:
                return memo[i]

            res = float('inf')
            for word, cost in zip(words, costs):
                if target[i:].startswith(word):
                    res = min(res, cost + dp(i + len(word)))

            memo[i] = res if res != float('inf') else -1
            return memo[i]

        result = dp(0)
        return result if result != float('inf') else -1