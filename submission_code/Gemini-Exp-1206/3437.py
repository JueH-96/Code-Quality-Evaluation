from typing import List
from collections import defaultdict

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        power_counts = defaultdict(int)
        for p in power:
            power_counts[p] += 1
        
        unique_powers = sorted(power_counts.keys())
        n = len(unique_powers)
        
        dp = {}
        
        def solve(index):
            if index >= n:
                return 0
            
            if index in dp:
                return dp[index]
            
            current_power = unique_powers[index]
            
            # Option 1: Don't cast the current spell
            option1 = solve(index + 1)
            
            # Option 2: Cast the current spell
            option2 = current_power * power_counts[current_power]
            
            next_index = index + 1
            while next_index < n and unique_powers[next_index] <= current_power + 2:
                next_index += 1
            
            option2 += solve(next_index)
            
            dp[index] = max(option1, option2)
            return dp[index]
        
        return solve(0)