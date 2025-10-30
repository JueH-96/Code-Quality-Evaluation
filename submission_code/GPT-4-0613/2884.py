class Solution:
    def longestValidSubstring(self, word: str, forbidden: list[str]) -> int:
        forbidden = set(forbidden)
        n = len(word)
        dp = [0] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = dp[i - 1] + 1
            for j in range(i):
                if word[j:i] in forbidden:
                    dp[i] = min(dp[i], dp[j])
        return max(dp)