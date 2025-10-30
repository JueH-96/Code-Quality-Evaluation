class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        n = len(s)
        dp = [0] * (n + 1)
        count = [0] * 26
        for i in range(n - 1, -1, -1):
            count[ord(s[i]) - ord('a')] += 1
            dp[i] = dp[i + 1]
            for j in range(k):
                dp[i] = max(dp[i], dp[i + 1 if count[j] == 0 else i] + 1)
            count[ord(s[i]) - ord('a')] -= 1
        return dp[0]