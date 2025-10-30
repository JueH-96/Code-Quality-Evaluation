from typing import List
import math

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        from math import gcd
        
        def lcm(a: int, b: int) -> int:
            return a * b // gcd(a, b)
        
        def score(arr: List[int]) -> int:
            if not arr:
                return 0
            g = arr[0]
            l = arr[0]
            for x in arr[1:]:
                g = gcd(g, x)
                l = lcm(l, x)
            return g * l
        
        best = score(nums)
        n = len(nums)
        for i in range(n):
            # remove nums[i]
            curr = score(nums[:i] + nums[i+1:])
            if curr > best:
                best = curr
        
        return best