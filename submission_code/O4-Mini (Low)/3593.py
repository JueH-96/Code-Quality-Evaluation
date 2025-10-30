from typing import List
import math

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        # Helper to compute GCD of a list
        def gcd_list(arr: List[int]) -> int:
            g = 0
            for x in arr:
                g = math.gcd(g, x)
            return g
        
        # Helper to compute LCM of a list
        def lcm_list(arr: List[int]) -> int:
            l = 1
            for x in arr:
                l = l * x // math.gcd(l, x)
            return l
        
        n = len(nums)
        best = 0
        
        # Try without removing any element
        if n > 0:
            g = gcd_list(nums)
            l = lcm_list(nums)
            best = max(best, g * l)
        
        # Try removing each element
        for i in range(n):
            arr = nums[:i] + nums[i+1:]
            if not arr:
                # Empty array score is defined as 0
                continue
            g = gcd_list(arr)
            l = lcm_list(arr)
            best = max(best, g * l)
        
        return best