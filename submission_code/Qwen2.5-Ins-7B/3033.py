class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        diff = [(i, s1[i] != s2[i]) for i in range(n) if s1[i] != s2[i]]
        if not diff:
            return 0
        if len(diff) % 2 == 1:
            return -1
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        for i in range(1, n + 1):
            if i < n - 1 and diff[i-1][1]:
                dp[i] = min(dp[i], dp[i-1] + 1)
            if i >= 2 and diff[i-2][1] and (i - 2) * x <= dp[i-2]:
                dp[i] = min(dp[i], dp[i-2] + x)
            if i >= 2 and diff[i-1][1] and diff[i-2][1]:
                dp[i] = min(dp[i], dp[i-2] + 1)
        return dp[n] if dp[n] != float('inf') else -1