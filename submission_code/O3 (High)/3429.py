from typing import List

class Solution:
    def isArraySpecial(self, nums: List[int]) -> bool:
        """
        An array is 'special' if every pair of adjacent elements has different parity.
        This method returns True when `nums` is special, otherwise False.
        """
        # Traverse each adjacent pair and compare their parity.
        for i in range(1, len(nums)):
            # If both numbers share the same parity (both even or both odd), return False.
            if (nums[i] & 1) == (nums[i - 1] & 1):
                return False

        # If no adjacent elements have the same parity, the array is special.
        return True