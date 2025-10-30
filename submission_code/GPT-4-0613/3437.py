class Solution:
    def maximumTotalDamage(self, power):
        power.sort()
        dp = [0] * len(power)
        dp[0] = power[0]
        for i in range(1, len(power)):
            dp[i] = max(dp[i-1], power[i] + (dp[i-2] if i > 1 else 0))
        return dp[-1]