class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        word_set = {word: cost for word, cost in zip(words, costs)}
        for i in range(1, n + 1):
            for j in range(i):
                if target[j:i] in word_set:
                    dp[i] = min(dp[i], dp[j] + word_set[target[j:i]])
        return dp[-1] if dp[-1] != float('inf') else -1