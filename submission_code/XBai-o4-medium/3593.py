import math
from typing import List

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        def compute(arr: List[int]) -> int:
            if not arr:
                return 0
            if len(arr) == 1:
                return arr[0] * arr[0]
            gcd_val = arr[0]
            lcm_val = arr[0]
            for num in arr[1:]:
                gcd_val = math.gcd(gcd_val, num)
                lcm_val = (lcm_val * num) // math.gcd(lcm_val, num)
            return gcd_val * lcm_val
        
        max_score = compute(nums)
        n = len(nums)
        for i in range(n):
            new_arr = nums[:i] + nums[i+1:]
            current = compute(new_arr)
            if current > max_score:
                max_score = current
        return max_score