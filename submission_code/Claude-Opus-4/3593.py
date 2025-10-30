class Solution:
    def maxScore(self, nums: List[int]) -> int:
        from math import gcd
        from functools import reduce
        
        def gcd_of_list(lst):
            if not lst:
                return 0
            return reduce(gcd, lst)
        
        def lcm(a, b):
            if a == 0 or b == 0:
                return 0
            return abs(a * b) // gcd(a, b)
        
        def lcm_of_list(lst):
            if not lst:
                return 0
            return reduce(lcm, lst)
        
        # If array has only one element
        if len(nums) == 1:
            return nums[0] * nums[0]
        
        # Calculate factor score without removing any element
        max_score = gcd_of_list(nums) * lcm_of_list(nums)
        
        # Try removing each element one by one
        for i in range(len(nums)):
            # Create array without element at index i
            remaining = nums[:i] + nums[i+1:]
            if remaining:  # Make sure we have elements left
                current_gcd = gcd_of_list(remaining)
                current_lcm = lcm_of_list(remaining)
                current_score = current_gcd * current_lcm
                max_score = max(max_score, current_score)
        
        return max_score