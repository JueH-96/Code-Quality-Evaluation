from typing import List

class Solution:
    def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
        n = len(damage)
        dp = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            time_to_die = (health[i] + damage[i] - 1) // damage[i]
            if power >= health[i]:
                dp[i] = max(dp[i + 1], health[i])
            else:
                dp[i] = min((time_to_die - 1) * power + dp[i + time_to_die], (time_to_die) * power + max(0, health[i] - time_to_die * damage[i]) + dp[i + time_to_die])
        return dp[0]