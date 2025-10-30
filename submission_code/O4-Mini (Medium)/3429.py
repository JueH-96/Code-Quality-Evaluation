from typing import List

class Solution:
    def isArraySpecial(self, nums: List[int]) -> bool:
        # Check each adjacent pair for differing parity
        for i in range(len(nums) - 1):
            if nums[i] % 2 == nums[i + 1] % 2:
                return False
        return True