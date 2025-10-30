class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        counts = {}
        for p in power:
            counts[p] = counts.get(p, 0) + 1

        unique_powers = sorted(counts.keys())
        n = len(unique_powers)
        dp = [0] * n

        for i in range(n):
            current_power = unique_powers[i]
            current_count = counts[current_power]
            damage_with_current = current_count * current_power

            max_damage_before = 0
            for j in range(i):
                if unique_powers[j] < current_power - 2:
                    max_damage_before = max(max_damage_before, dp[j])

            damage_if_use_current = damage_with_current + max_damage_before

            damage_if_not_use_current = dp[i - 1] if i > 0 else 0

            dp[i] = max(damage_if_not_use_current, damage_if_use_current)

        return dp[n - 1] if n > 0 else 0