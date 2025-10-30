class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        n = len(damage)
        dp = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            dp[i] = max(0, dp[i + 1]) + damage[i]
            if health[i] <= power:
                power -= health[i]
            else:
                dp[i] -= min(power, damage[i])
        return dp[0]