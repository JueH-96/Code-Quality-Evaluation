from typing import List

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0

        word_cost = {}
        for word, cost in zip(words, costs):
            if word in word_cost:
                word_cost[word] = min(word_cost[word], cost)
            else:
                word_cost[word] = cost

        for i in range(1, n + 1):
            for word, cost in word_cost.items():
                word_len = len(word)
                if i >= word_len and target[:i].endswith(word):
                    dp[i] = min(dp[i], dp[i - word_len] + cost)

        return dp[n] if dp[n] != float('inf') else -1