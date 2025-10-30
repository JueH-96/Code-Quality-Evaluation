from typing import List
import collections

class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        prefix_sum = [0]
        for num in nums:
            prefix_sum.append(prefix_sum[-1] + num)
        
        num_to_min_prefix_sum = collections.defaultdict(lambda: float('inf'))
        max_sum = float('-inf')
        
        for i, num in enumerate(nums):
            for target in [num + k, num - k]:
                if target in num_to_min_prefix_sum:
                    max_sum = max(max_sum, prefix_sum[i + 1] - num_to_min_prefix_sum[target])
            num_to_min_prefix_sum[num] = min(num_to_min_prefix_sum[num], prefix_sum[i])
        
        return max_sum if max_sum > float('-inf') else 0