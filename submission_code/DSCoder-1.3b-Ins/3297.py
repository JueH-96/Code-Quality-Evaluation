class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        dp = [0] * (n + 1)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + ord(word[i])
            dp[i + 1] = dp[i] + ord(word[i])
        res = dp[n]
        for i in range(k, n + 1):
            res = min(res, dp[i] - prefix_sum[i - k] + prefix_sum[k] + k)
        return res