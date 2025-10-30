from typing import List

class Solution:
    def countSubarrays(self, nums: List[int]) -> int:
        count = 0
        for i in range(len(nums) - 2):
            x = nums[i]
            y = nums[i + 1]
            z = nums[i + 2]
            if x + z == y / 2:
                count += 1
        return count