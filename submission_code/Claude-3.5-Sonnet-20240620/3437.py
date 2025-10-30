class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        if not power:
            return 0
        
        damage_count = {}
        for p in power:
            damage_count[p] = damage_count.get(p, 0) + 1
        
        max_damage = max(power)
        dp = [0] * (max_damage + 1)
        
        for i in range(max_damage + 1):
            if i > 2:
                dp[i] = dp[i-3]
            dp[i] += i * damage_count.get(i, 0)
            
            if i > 0:
                dp[i] = max(dp[i], dp[i-1])
        
        return dp[max_damage]