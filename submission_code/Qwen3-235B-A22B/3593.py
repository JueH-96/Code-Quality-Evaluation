import math
from typing import List

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0  # As per problem constraints, n >= 1
        
        # Initialize prefix and suffix arrays
        prefix_gcd = [0] * n
        prefix_lcm = [0] * n
        suffix_gcd = [0] * n
        suffix_lcm = [0] * n
        
        # Build prefix arrays
        prefix_gcd[0] = nums[0]
        prefix_lcm[0] = nums[0]
        for i in range(1, n):
            prefix_gcd[i] = math.gcd(prefix_gcd[i-1], nums[i])
            prefix_lcm[i] = (prefix_lcm[i-1] * nums[i]) // math.gcd(prefix_lcm[i-1], nums[i])
        
        # Build suffix arrays
        suffix_gcd[-1] = nums[-1]
        suffix_lcm[-1] = nums[-1]
        for i in range(n-2, -1, -1):
            suffix_gcd[i] = math.gcd(suffix_gcd[i+1], nums[i])
            suffix_lcm[i] = (suffix_lcm[i+1] * nums[i]) // math.gcd(suffix_lcm[i+1], nums[i])
        
        # Calculate the original score without any removal
        max_score = prefix_gcd[-1] * prefix_lcm[-1]
        
        # Iterate over each element to consider removing it
        for i in range(n):
            if n == 1:
                current_product = 0
            else:
                if i == 0:
                    gcd_val = suffix_gcd[1]
                    lcm_val = suffix_lcm[1]
                elif i == n - 1:
                    gcd_val = prefix_gcd[n-2]
                    lcm_val = prefix_lcm[n-2]
                else:
                    gcd_val = math.gcd(prefix_gcd[i-1], suffix_gcd[i+1])
                    lcm_val = (prefix_lcm[i-1] * suffix_lcm[i+1]) // math.gcd(prefix_lcm[i-1], suffix_lcm[i+1])
                current_product = gcd_val * lcm_val
            max_score = max(max_score, current_product)
        
        return max_score