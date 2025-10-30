from typing import List

class Solution:
    def isArraySpecial(self, nums: List[int]) -> bool:
        # Iterate through the list checking adjacent elements
        for i in range(len(nums) - 1):
            # If both adjacent numbers have the same remainder when divided by 2,
            # then they are either both even or both odd.
            if nums[i] % 2 == nums[i+1] % 2:
                return False
        return True