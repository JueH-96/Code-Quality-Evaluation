class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        dp = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            dp[i] = min(dp[j] + (n - j) for j in range(i + k, n + 1)) + 1
        return dp[0]