class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        power.sort()
        n = len(power)
        dp = [0] * n
        dp[0] = power[0]
        for i in range(1, n):
            dp[i] = power[i]
            if i >= 3:
                dp[i] += max(dp[i-3], dp[i-4])
            elif i >= 2:
                dp[i] += dp[i-2]
            elif i >= 1:
                dp[i] += dp[i-1]
        return max(dp)