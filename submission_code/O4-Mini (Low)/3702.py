from typing import List
import math

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        n = len(nums)
        best = 1
        
        for i in range(n):
            g = nums[i]
            l = nums[i]
            prod = nums[i]
            # single element subarray always satisfies prod = a, gcd = a, lcm = a -> a = a * a? 
            # Actually only holds when a = 1. But we only update best on valid subarrays below.
            if prod == g * l:
                best = max(best, 1)
                
            for j in range(i+1, n):
                x = nums[j]
                # update gcd, lcm, product
                g = math.gcd(g, x)
                l = l * x // math.gcd(l, x)
                prod *= x
                # check product-equivalence
                if prod == g * l:
                    best = max(best, j - i + 1)
        
        return best