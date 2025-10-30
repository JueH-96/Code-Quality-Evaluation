import math
from typing import List

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        n = len(nums)
        prefix_gcd = [0] * n
        prefix_lcm = [0] * n
        suffix_gcd = [0] * n
        suffix_lcm = [0] * n
        
        prefix_gcd[0] = nums[0]
        prefix_lcm[0] = nums[0]
        for i in range(1, n):
            prefix_gcd[i] = math.gcd(prefix_gcd[i-1], nums[i])
            prefix_lcm[i] = prefix_lcm[i-1] * nums[i] // math.gcd(prefix_lcm[i-1], nums[i])
        
        suffix_gcd[n-1] = nums[n-1]
        suffix_lcm[n-1] = nums[n-1]
        for i in range(n-2, -1, -1):
            suffix_gcd[i] = math.gcd(suffix_gcd[i+1], nums[i])
            suffix_lcm[i] = suffix_lcm[i+1] * nums[i] // math.gcd(suffix_lcm[i+1], nums[i])
        
        max_score = prefix_gcd[-1] * prefix_lcm[-1]
        
        for i in range(n):
            if n == 1:
                score = 0
            else:
                if i == 0:
                    gcd_val = suffix_gcd[1]
                    lcm_val = suffix_lcm[1]
                elif i == n-1:
                    gcd_val = prefix_gcd[n-2]
                    lcm_val = prefix_lcm[n-2]
                else:
                    gcd_val = math.gcd(prefix_gcd[i-1], suffix_gcd[i+1])
                    lcm_val = prefix_lcm[i-1] * suffix_lcm[i+1] // math.gcd(prefix_lcm[i-1], suffix_lcm[i+1])
                score = gcd_val * lcm_val
            if score > max_score:
                max_score = score
        return max_score