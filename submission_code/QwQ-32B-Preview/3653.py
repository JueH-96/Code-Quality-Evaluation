from typing import List
import math

class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        group_min = [math.inf] * k
        group_min[0] = 0  # prefix[0] is 0
        
        max_sum = -math.inf
        
        for j in range(n):
            r = (j + 1) % k
            if group_min[r] != math.inf:
                current_sum = prefix[j + 1] - group_min[r]
                if current_sum > max_sum:
                    max_sum = current_sum
            # Update the group for the current index j
            group_index = j % k
            group_min[group_index] = min(group_min[group_index], prefix[j + 1])
        
        return max_sum