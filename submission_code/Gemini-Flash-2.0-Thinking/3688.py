from typing import List

class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        def kadane(arr):
            if not arr:
                return -float('inf')
            current_max = 0
            global_max = -float('inf')
            for x in arr:
                current_max = max(x, current_max + x)
                global_max = max(global_max, current_max)
            return global_max

        max_so_far = kadane(nums)
        unique_elements = set(nums)

        for x in unique_elements:
            modified_nums = [val for val in nums if val != x]
            if modified_nums:
                max_so_far = max(max_so_far, kadane(modified_nums))

        return max_so_far