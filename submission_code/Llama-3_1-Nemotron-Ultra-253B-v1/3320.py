from typing import List

class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        if len(nums) < 2:
            return 0
        s = nums[0] + nums[1]
        count = 0
        i = 0
        while i + 1 < len(nums):
            if nums[i] + nums[i+1] == s:
                count += 1
                i += 2
            else:
                break
        return count