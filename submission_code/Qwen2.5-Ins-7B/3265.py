from typing import List
from collections import defaultdict

class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        prefix_sum = 0
        min_prefix_sum = 0
        min_prefix_sum_map = defaultdict(lambda: float('inf'))
        max_sum = float('-inf')
        
        for i, num in enumerate(nums):
            prefix_sum += num
            if i > 0:
                min_prefix_sum = min(min_prefix_sum, prefix_sum - nums[i-1])
            if num - k in min_prefix_sum_map:
                max_sum = max(max_sum, prefix_sum - min_prefix_sum_map[num - k])
            if num + k in min_prefix_sum_map:
                max_sum = max(max_sum, prefix_sum - min_prefix_sum_map[num + k])
            min_prefix_sum_map[prefix_sum - num] = min(min_prefix_sum_map[prefix_sum - num], min_prefix_sum)
        
        return max_sum if max_sum != float('-inf') else 0