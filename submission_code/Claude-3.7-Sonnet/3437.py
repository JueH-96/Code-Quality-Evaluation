from collections import Counter

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        # Group spells by damage values
        damage_freq = Counter(power)
        damage_values = sorted(damage_freq.keys())
        
        # Initialize DP array
        n = len(damage_values)
        if n == 0:
            return 0
        
        dp = [0] * n
        dp[0] = damage_values[0] * damage_freq[damage_values[0]]
        
        idx = 0  # Tracks the first index where damage_values[idx] + 2 >= current_damage
        max_dp_up_to = 0  # Maximum dp value for all j such that j < idx
        
        for i in range(1, n):
            current_damage = damage_values[i]
            current_total = current_damage * damage_freq[current_damage]
            
            # Update idx and max_dp_up_to
            while idx < i and damage_values[idx] + 2 < current_damage:
                max_dp_up_to = max(max_dp_up_to, dp[idx])
                idx += 1
            
            # Skip current damage value
            skip_damage = dp[i-1]
            
            # Include current damage value
            include_damage = max_dp_up_to + current_total
            
            dp[i] = max(skip_damage, include_damage)
        
        return dp[n-1]