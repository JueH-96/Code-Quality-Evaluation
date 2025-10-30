import math
from typing import List

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        def factor_score(subset):
            if not subset:
                return 0
            gcd = math.gcd(*subset)
            lcm = math.lcm(*subset)
            return gcd * lcm
        
        # Calculate factor score of the original array
        original_score = factor_score(nums)
        max_score = original_score
        
        # Iterate through each element and calculate score without it
        for i in range(len(nums)):
            subset = nums[:i] + nums[i+1:]
            score = factor_score(subset)
            if score > max_score:
                max_score = score
                
        return max_score