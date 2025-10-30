import math
from typing import List

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        def factor_score(arr):
            if not arr:
                return 0
            # Compute GCD
            gcd_val = arr[0]
            for num in arr[1:]:
                gcd_val = math.gcd(gcd_val, num)
            # Compute LCM
            lcm_val = 1
            for num in arr:
                lcm_val = (lcm_val * num) // math.gcd(lcm_val, num)
            return gcd_val * lcm_val
        
        # Initialize max_score with no removal case
        max_score = factor_score(nums)
        
        # Check score after removing each element
        for i in range(len(nums)):
            # Create sublist without the i-th element
            sub_nums = nums[:i] + nums[i+1:]
            # Update max_score with the score after removal
            max_score = max(max_score, factor_score(sub_nums))
        
        return max_score