class Solution:
    def maxLength(self, nums: List[int]) -> int:
        from math import gcd
        from functools import reduce
        
        def array_gcd(arr):
            return reduce(gcd, arr)
        
        def array_lcm(arr):
            def lcm(a, b):
                return abs(a * b) // gcd(a, b)
            return reduce(lcm, arr)
        
        def array_product(arr):
            result = 1
            for num in arr:
                result *= num
            return result
        
        n = len(nums)
        max_length = 0
        
        # Check all possible subarrays
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                
                # Calculate product, gcd, and lcm
                prod = array_product(subarray)
                g = array_gcd(subarray)
                l = array_lcm(subarray)
                
                # Check if product equivalent
                if prod == g * l:
                    max_length = max(max_length, j - i + 1)
        
        return max_length