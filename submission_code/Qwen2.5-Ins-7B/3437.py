from typing import List

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        power.sort()
        n = len(power)
        dp = [0] * n
        dp[0] = power[0]
        for i in range(1, n):
            dp[i] = power[i]
            for j in range(i-1, -1, -1):
                if power[i] - power[j] > 2:
                    break
                dp[i] = max(dp[i], dp[j] + power[i])
        return max(dp)