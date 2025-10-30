class Solution:
    def maxScore(self, nums: List[int]) -> int:
        import math
        from functools import reduce
        
        def gcd_of_list(lst):
            if not lst:
                return 0
            return reduce(math.gcd, lst)
        
        def lcm_of_two(a, b):
            return (a * b) // math.gcd(a, b)
        
        def lcm_of_list(lst):
            if not lst:
                return 0
            return reduce(lcm_of_two, lst)
        
        n = len(nums)
        
        # Special case: single element
        if n == 1:
            return nums[0] * nums[0]
        
        # Calculate score without removing any element
        max_score = gcd_of_list(nums) * lcm_of_list(nums)
        
        # Try removing each element
        for i in range(n):
            remaining = nums[:i] + nums[i+1:]
            if remaining:  # If there are elements left
                score = gcd_of_list(remaining) * lcm_of_list(remaining)
                max_score = max(max_score, score)
        
        return max_score