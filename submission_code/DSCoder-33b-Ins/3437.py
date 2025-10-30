from typing import List

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        power.sort()
        dp = [0] * len(power)
        dp[0] = power[0]
        for i in range(1, len(power)):
            if power[i] - power[i - 1] < 2:
                dp[i] = max(dp[i - 1], dp[i - 2] if i > 1 else 0, dp[i - 3] + power[i] if i > 2 else power[i])
            else:
                dp[i] = dp[i - 1] + power[i]
        return dp[-1]