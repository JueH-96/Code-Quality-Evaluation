class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        dp = [0] * (n + 1)
        for i in range(k - 1, n):
            dp[i + 1] = dp[i]
            if s[i] != s[i - k + 1]:
                dp[i + 1] += 1
        dp2 = [0] * (n + 1)
        for i in range(k - 1, n):
            dp2[i + 1] = min(dp2[i], dp2[i - k + 1])
            if s[i] != s[i - k + 1]:
                dp2[i + 1] += 1
        return min(dp[n], dp2[n])