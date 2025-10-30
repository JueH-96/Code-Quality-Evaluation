from math import gcd
from functools import reduce

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        def compute_gcd(arr):
            return reduce(gcd, arr)
        
        def compute_lcm(arr):
            def lcm(a, b):
                return a * b // gcd(a, b)
            return reduce(lcm, arr)
        
        def compute_prod(arr):
            return reduce(lambda x, y: x * y, arr)
        
        max_len = 0
        n = len(nums)
        for i in range(n):
            current_gcd = nums[i]
            current_lcm = nums[i]
            current_prod = nums[i]
            for j in range(i, n):
                current_gcd = gcd(current_gcd, nums[j])
                current_lcm = current_lcm * nums[j] // gcd(current_lcm, nums[j])
                current_prod *= nums[j]
                if current_prod == current_lcm * current_gcd:
                    max_len = max(max_len, j - i + 1)
        return max_len