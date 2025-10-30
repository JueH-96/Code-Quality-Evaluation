from typing import List

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        power_counts = {}
        for p in power:
            power_counts[p] = power_counts.get(p, 0) + 1
        sorted_powers = sorted(list(power_counts.keys()))
        dp = {}
        dp[None] = 0
        if not sorted_powers:
            return 0
        dp[sorted_powers[0]] = sorted_powers[0] * power_counts[sorted_powers[0]]
        for i in range(1, len(sorted_powers)):
            current_power = sorted_powers[i]
            current_damage = current_power * power_counts[current_power]
            prev_valid_power = None
            for j in range(i - 1, -1, -1):
                if sorted_powers[j] < current_power - 2:
                    prev_valid_power = sorted_powers[j]
                    break
            damage_with_current = current_damage + dp.get(prev_valid_power, 0)
            damage_without_current = dp.get(sorted_powers[i-1], 0)
            dp[current_power] = max(damage_with_current, damage_without_current)
        return dp[sorted_powers[-1]] if sorted_powers else 0