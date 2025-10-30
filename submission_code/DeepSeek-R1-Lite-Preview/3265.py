from typing import List
from math import inf

class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i+1] = prefix_sum[i] + nums[i]
        
        num_to_min_prefix = {}
        max_sum = -inf
        
        for j in range(n):
            target1 = nums[j] + k
            target2 = nums[j] - k
            
            if target1 in num_to_min_prefix:
                current_sum = prefix_sum[j+1] - num_to_min_prefix[target1]
                if current_sum > max_sum:
                    max_sum = current_sum
            if target2 in num_to_min_prefix:
                current_sum = prefix_sum[j+1] - num_to_min_prefix[target2]
                if current_sum > max_sum:
                    max_sum = current_sum
            
            if nums[j] in num_to_min_prefix:
                num_to_min_prefix[nums[j]] = min(num_to_min_prefix[nums[j]], prefix_sum[j])
            else:
                num_to_min_prefix[nums[j]] = prefix_sum[j]
        
        return max_sum if max_sum != -inf else 0