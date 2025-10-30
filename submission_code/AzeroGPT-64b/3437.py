from collections import Counter

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        counts = Counter(power)
        sorted_power = sorted(counts.keys())
        
        dp = [0] * (len(sorted_power) + 2)
        dp[0] = 0
        dp[1] = sorted_power[0] * counts[sorted_power[0]]
        exclude = 0
        
        for i in range(1, len(sorted_power)):
            current_power = sorted_power[i]
            last_power = sorted_power[i-1]
            
            # Case 1: Include current power, exclude neighbors
            if last_power + 1 < current_power - 1:
                include = max(dp[i - 1], exclude) + current_power * counts[current_power]
            else:
                include = exclude + current_power * counts[current_power]
            
            # Case 2: Exclude current power
            exclude = max(dp[i - 1], exclude)
            
            dp[i + 1] = max(include, exclude)
        
        return max(dp[-1], dp[-2])