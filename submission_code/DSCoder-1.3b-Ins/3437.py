class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        n = len(power)
        max_damage = [0] * (n + 2)
        for i in range(n - 1, -1, -1):
            max_damage[i] = min(max_damage[i + 1], power[i])
        return max(max_damage)