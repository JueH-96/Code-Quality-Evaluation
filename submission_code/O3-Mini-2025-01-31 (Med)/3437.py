from typing import List
import bisect

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        # Count frequencies of each damage value and compute total damage contribution
        freq = {}
        for p in power:
            freq[p] = freq.get(p, 0) + 1
        
        # Create sorted list of distinct damage values
        # For each damage value d, the "value" is d multiplied by its frequency.
        distinct = sorted(freq.keys())
        values = [d * freq[d] for d in distinct]
        
        # dp[i] stores the maximum total damage achievable considering distinct[0..i]
        dp = [0] * len(distinct)
        dp[0] = values[0]
        
        for i in range(1, len(distinct)):
            # Option 1: skip the current damage value
            option_skip = dp[i-1]
            
            # Option 2: take the current damage value, add to the best we can achieve with a damage value <= current - 3.
            # We need to find the rightmost index j such that distinct[j] <= distinct[i] - 3.
            current_damage = distinct[i]
            # We want to search for current_damage - 3; using bisect_right gives first index > (current_damage - 3)
            j = bisect.bisect_right(distinct, current_damage - 3) - 1
            
            if j >= 0:
                take_value = values[i] + dp[j]
            else:
                take_value = values[i]
            
            dp[i] = max(option_skip, take_value)
        
        return dp[-1]