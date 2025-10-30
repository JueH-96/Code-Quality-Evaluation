class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        for i in range(n):
            for j, word in enumerate(words):
                if target.startswith(word, i):
                    length = len(word)
                    dp[i + length] = min(dp[i + length], dp[i] + costs[j])
        return dp[n] if dp[n] != float('inf') else -1