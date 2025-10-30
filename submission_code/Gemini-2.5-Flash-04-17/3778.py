from typing import List

class Solution:
    def transformArray(self, nums: List[int]) -> List[int]:
        # Step 1: Replace each even number with 0.
        # Step 2: Replace each odd numbers with 1.
        # We can do this in a single pass by checking the parity.
        # Iterate through the array and modify elements in place.
        for i in range(len(nums)):
            if nums[i] % 2 == 0:
                # If the number is even, replace it with 0.
                nums[i] = 0
            else:
                # If the number is odd, replace it with 1.
                nums[i] = 1
        
        # Step 3: Sort the modified array in non-decreasing order.
        # The array now contains only 0s and 1s. Sorting will place
        # all the 0s before all the 1s.
        nums.sort()
        
        # Return the resulting array after performing all operations.
        return nums