class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        dp = [0] * (x + y + z + 3)
        dp[2], dp[3] = 2, 3
        for i in range(4, x + y + z + 3):
            dp[i] = max(dp[i - 2], dp[i - 3]) + 2
        return max(dp[x + y + z], dp[x + y], dp[x + z], dp[y + z])