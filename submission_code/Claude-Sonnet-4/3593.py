class Solution:
    def maxScore(self, nums: List[int]) -> int:
        from math import gcd
        from functools import reduce
        
        def lcm(a, b):
            return a * b // gcd(a, b)
        
        def array_gcd(arr):
            if not arr:
                return 0
            return reduce(gcd, arr)
        
        def array_lcm(arr):
            if not arr:
                return 0
            return reduce(lcm, arr)
        
        def factor_score(arr):
            if not arr:
                return 0
            if len(arr) == 1:
                return arr[0] * arr[0]  # GCD and LCM of single number is the number itself
            return array_gcd(arr) * array_lcm(arr)
        
        n = len(nums)
        max_score = 0
        
        # Try original array (removing 0 elements)
        max_score = max(max_score, factor_score(nums))
        
        # Try removing each element one by one
        for i in range(n):
            # Create array without element at index i
            new_array = nums[:i] + nums[i+1:]
            max_score = max(max_score, factor_score(new_array))
        
        return max_score