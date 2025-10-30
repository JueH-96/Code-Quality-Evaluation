class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        if not power:
            return 0
        
        power.sort()
        
        n = len(power)
        
        dp = [0] * n
        
        dp[0] = power[0]
        
        for i in range(1, n):
            
            max_damage_without_current = dp[i-1] if i > 0 else 0
            
            max_damage_with_current = power[i]
            
            j = i - 1
            while j >= 0 and power[i] - power[j] <= 2:
                j -= 1
            
            if j >= 0:
                max_damage_with_current += dp[j]
            
            dp[i] = max(max_damage_without_current, max_damage_with_current)
        
        return dp[n-1]