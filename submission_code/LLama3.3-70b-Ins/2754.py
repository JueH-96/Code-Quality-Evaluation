from typing import List
import itertools

class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        max_strength = float('-inf')
        
        # Generate all possible subsets of the given list
        for r in range(1, len(nums) + 1):
            for subset in itertools.combinations(nums, r):
                # Calculate the strength of the current subset
                strength = 1
                for num in subset:
                    strength *= num
                
                # Update the maximum strength if the current strength is greater
                max_strength = max(max_strength, strength)
        
        return max_strength