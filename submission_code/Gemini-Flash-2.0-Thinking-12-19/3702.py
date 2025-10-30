import math
from typing import List

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        max_length = 0
        n = len(nums)
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                if not subarray:
                    continue
                product = 1
                for x in subarray:
                    product *= x
                current_gcd = subarray[0]
                for k in range(1, len(subarray)):
                    current_gcd = math.gcd(current_gcd, subarray[k])
                current_lcm = subarray[0]
                for k in range(1, len(subarray)):
                    current_lcm = (current_lcm * subarray[k]) // math.gcd(current_lcm, subarray[k])
                if product == current_lcm * current_gcd:
                    current_length = len(subarray)
                    max_length = max(max_length, current_length)
        return max_length