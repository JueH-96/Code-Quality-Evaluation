class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        f = set(forbidden)
        n = len(word)
        dp = [0] * n
        max_len = 0
        for i in range(n-1, -1, -1):
            dp[i] = 1
            for j in range(i+1, min(i+11, n)):
                if word[i:j+1] not in f:
                    dp[i] = max(dp[i], dp[j] + j - i)
            max_len = max(max_len, dp[i])
        return max_len