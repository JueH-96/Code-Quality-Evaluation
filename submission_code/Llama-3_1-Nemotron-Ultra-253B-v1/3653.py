from typing import List

class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        prefix_sum = 0
        min_prefix = {0: 0}
        max_sum = -float('inf')
        
        for i in range(1, len(nums) + 1):
            prefix_sum += nums[i - 1]
            r = i % k
            
            if r in min_prefix:
                current_sum = prefix_sum - min_prefix[r]
                if current_sum > max_sum:
                    max_sum = current_sum
            
            if r not in min_prefix or prefix_sum < min_prefix[r]:
                min_prefix[r] = prefix_sum
        
        return max_sum