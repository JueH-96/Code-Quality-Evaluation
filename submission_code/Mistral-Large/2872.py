from typing import List

class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        # Start from the end of the array
        for i in range(len(nums) - 2, -1, -1):
            if nums[i] <= nums[i + 1]:
                nums[i + 1] += nums[i]
            else:
                nums[i + 1] = nums[i]

        # The largest element will be the last element in the array
        return nums[-1]