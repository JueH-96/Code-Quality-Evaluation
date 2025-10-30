from typing import List

class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        prefix_sum = 0
        min_prefix_sum = {nums[0]: 0}
        max_sum = float('-inf')
        
        for i, num in enumerate(nums):
            prefix_sum += num
            
            if (num + k) in min_prefix_sum:
                max_sum = max(max_sum, prefix_sum - min_prefix_sum[num + k])
            if (num - k) in min_prefix_sum:
                max_sum = max(max_sum, prefix_sum - min_prefix_sum[num - k])
            
            if num in min_prefix_sum:
                min_prefix_sum[num] = min(min_prefix_sum[num], prefix_sum - num)
            else:
                min_prefix_sum[num] = prefix_sum - num
        
        return max_sum if max_sum != float('-inf') else 0