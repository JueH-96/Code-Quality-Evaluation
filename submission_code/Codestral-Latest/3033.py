class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        diff_positions = [i for i in range(n) if s1[i] != s2[i]]

        if len(diff_positions) % 2 != 0:
            return -1

        dp = [float('inf')] * (len(diff_positions) + 1)
        dp[0] = 0

        for i in range(1, len(diff_positions) + 1):
            for j in range(i):
                if (i - j) % 2 == 0:
                    cost = (diff_positions[i - 1] - diff_positions[j] + 1) * x if j < i - 1 else 0
                    dp[i] = min(dp[i], dp[j] + cost + (i - j - 1))

        return dp[len(diff_positions)] // 2