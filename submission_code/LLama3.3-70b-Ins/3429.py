from typing import List

class Solution:
    def isArraySpecial(self, nums: List[int]) -> bool:
        # Iterate over the array from the second element to the last element
        for i in range(1, len(nums)):
            # Check if the current element and the previous element have the same parity
            if nums[i] % 2 == nums[i-1] % 2:
                # If they have the same parity, return False
                return False
        # If no pairs with the same parity are found, return True
        return True