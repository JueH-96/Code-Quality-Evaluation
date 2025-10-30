from typing import List
import itertools

class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        total_power = 0
        
        # Generate all possible groups of heroes
        for r in range(1, len(nums) + 1):
            for group in itertools.combinations(nums, r):
                # Calculate the power of the current group
                max_val = max(group)
                min_val = min(group)
                power = (max_val ** 2) * min_val
                
                # Add the power of the current group to the total power
                total_power = (total_power + power) % MOD
        
        return total_power