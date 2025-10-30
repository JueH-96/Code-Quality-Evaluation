import math
from typing import List

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Compute original GCD and LCM
        original_gcd = nums[0]
        for num in nums[1:]:
            original_gcd = math.gcd(original_gcd, num)
        
        original_lcm = nums[0]
        for num in nums[1:]:
            original_lcm = original_lcm * num // math.gcd(original_lcm, num)
        
        max_score = original_gcd * original_lcm
        
        # Check all possible removals
        for i in range(n):
            new_nums = nums[:i] + nums[i+1:]
            if not new_nums:
                current_score = 0
            else:
                # Compute GCD of new_nums
                current_gcd = new_nums[0]
                for num in new_nums[1:]:
                    current_gcd = math.gcd(current_gcd, num)
                
                # Compute LCM of new_nums
                current_lcm = new_nums[0]
                for num in new_nums[1:]:
                    current_lcm = current_lcm * num // math.gcd(current_lcm, num)
                
                current_score = current_gcd * current_lcm
            
            if current_score > max_score:
                max_score = current_score
        
        return max_score