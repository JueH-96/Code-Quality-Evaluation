from typing import List

class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        if not nums:
            return 1  # If nums is empty, return 1 as the smallest missing positive integer

        # Find the longest sequential prefix
        sum_prefix = nums[0]
        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1] + 1:
                sum_prefix += nums[i]
            else:
                break

        nums_set = set(nums)
        x = sum_prefix
        while x in nums_set:
            x += 1
        return x