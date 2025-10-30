class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        dp = [0] + [float('inf')] * len(target)
        for i in range(len(target)):
            dp[i + 1] = min(dp[i + 1], dp[i] + costs[i])
            for word, cost in zip(words, costs):
                if target[i - len(word) + 1:i + 1] == word:
                    dp[i + 1] = min(dp[i + 1], dp[i - len(word) + 1] + cost)
        return dp[-1] if dp[-1] != float('inf') else -1