from typing import List

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        from collections import Counter
        
        # Count the frequency of each power value
        power_count = Counter(power)
        
        # Sort the unique power values
        unique_powers = sorted(power_count.keys())
        
        # Initialize dp array
        dp = [0] * (len(unique_powers) + 2)
        
        for i in range(len(unique_powers)):
            current_power = unique_powers[i]
            current_damage = current_power * power_count[current_power]
            
            # Check if we can take the current power without violating the constraint
            if i > 0 and unique_powers[i] == unique_powers[i-1] + 1:
                dp[i+1] = max(dp[i], dp[i-1] + current_damage)
            else:
                dp[i+1] = dp[i] + current_damage
        
        return dp[len(unique_powers)]