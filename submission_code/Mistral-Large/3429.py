from typing import List

class Solution:
    def isArraySpecial(self, nums: List[int]) -> bool:
        # If the array has only one element, it is considered special
        if len(nums) == 1:
            return True

        # Check the parity of adjacent elements
        for i in range(1, len(nums)):
            if (nums[i] % 2 == 0 and nums[i-1] % 2 == 0) or (nums[i] % 2 != 0 and nums[i-1] % 2 != 0):
                return False

        return True