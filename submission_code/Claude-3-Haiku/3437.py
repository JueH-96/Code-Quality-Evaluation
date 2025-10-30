class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        power.sort()
        n = len(power)
        dp = [0] * n
        
        for i in range(n):
            max_damage = 0
            for j in range(i):
                if abs(power[i] - power[j]) > 2:
                    max_damage = max(max_damage, dp[j] + power[i])
            for j in range(i+1, n):
                if abs(power[j] - power[i]) > 2:
                    max_damage = max(max_damage, dp[j] + power[i])
            dp[i] = max_damage
        
        return max(dp)