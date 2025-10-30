class Solution:
    def maxScore(self, nums: List[int]) -> int:
        from math import gcd
        from functools import reduce
        
        def lcm(a, b):
            return a * b // gcd(a, b)
        
        def array_gcd(arr):
            return reduce(gcd, arr)
        
        def array_lcm(arr):
            return reduce(lcm, arr)
        
        def factor_score(arr):
            if not arr:
                return 0
            return array_gcd(arr) * array_lcm(arr)
        
        # Try original array
        max_score = factor_score(nums)
        
        # Try removing each element
        for i in range(len(nums)):
            new_array = nums[:i] + nums[i+1:]
            max_score = max(max_score, factor_score(new_array))
        
        return max_score