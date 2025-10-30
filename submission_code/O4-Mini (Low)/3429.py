from typing import List

class Solution:
    def isArraySpecial(self, nums: List[int]) -> bool:
        # Check each adjacent pair
        for i in range(len(nums) - 1):
            # If both are even or both are odd, it's not special
            if nums[i] % 2 == nums[i + 1] % 2:
                return False
        return True