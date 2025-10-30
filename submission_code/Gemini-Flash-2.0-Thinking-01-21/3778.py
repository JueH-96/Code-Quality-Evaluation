from typing import List

class Solution:
    def transformArray(self, nums: List[int]) -> List[int]:
        # Step 1: Replace each even number with 0 and each odd number with 1.
        # Iterate through the array and modify elements in-place.
        for i in range(len(nums)):
            # Check if the number is even
            if nums[i] % 2 == 0:
                nums[i] = 0  # Replace even with 0
            else:
                nums[i] = 1  # Replace odd with 1

        # Step 2: Sort the modified array in non-decreasing order.
        # Since the array now only contains 0s and 1s, sorting will place all 0s before all 1s.
        nums.sort()

        # Return the resulting array after performing these operations.
        return nums