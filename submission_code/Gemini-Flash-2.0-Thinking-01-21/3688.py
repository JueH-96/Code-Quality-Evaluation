from typing import List

class Solution:
    def get_max_subarray_sum(self, arr: List[int]) -> int:
        if not arr:
            return -float('inf')
        max_so_far = -float('inf')
        current_max = 0
        for num in arr:
            current_max = max(num, current_max + num)
            max_so_far = max(max_so_far, current_max)
        return max_so_far

    def maxSubarraySum(self, nums: List[int]) -> int:
        unique_nums = set(nums)
        max_result = -float('inf')
        
        # Max subarray sum of original nums
        max_result = max(max_result, self.get_max_subarray_sum(nums))
        
        for x in unique_nums:
            temp_nums = [num for num in nums if num != x]
            current_max_sum = self.get_max_subarray_sum(temp_nums)
            max_result = max(max_result, current_max_sum)
            
        if max_result == -float('inf'):
            return 0
        return max_result