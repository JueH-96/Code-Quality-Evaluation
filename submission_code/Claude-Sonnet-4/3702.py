class Solution:
    def maxLength(self, nums: List[int]) -> int:
        from math import gcd
        from functools import reduce
        
        def lcm(a, b):
            return a * b // gcd(a, b)
        
        def array_gcd(arr):
            return reduce(gcd, arr)
        
        def array_lcm(arr):
            return reduce(lcm, arr)
        
        def array_product(arr):
            result = 1
            for x in arr:
                result *= x
            return result
        
        n = len(nums)
        max_len = 0
        
        # Check all possible subarrays
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                
                prod = array_product(subarray)
                arr_gcd = array_gcd(subarray)
                arr_lcm = array_lcm(subarray)
                
                if prod == arr_lcm * arr_gcd:
                    max_len = max(max_len, len(subarray))
        
        return max_len