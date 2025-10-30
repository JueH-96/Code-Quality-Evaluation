class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        power.sort()
        n = len(power)
        dp = [0] * n
        dp[0] = power[0]
        for i in range(1, n):
            dp[i] = max(dp[i-1], dp[i-2] + power[i])
        return dp[-1]