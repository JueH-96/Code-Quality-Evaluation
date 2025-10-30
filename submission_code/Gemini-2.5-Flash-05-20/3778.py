from typing import List

class Solution:
    def transformArray(self, nums: List[int]) -> List[int]:
        # Step 1 & 2: Replace each even number with 0 and each odd number with 1.
        # Iterate through the array and modify elements in-place.
        for i in range(len(nums)):
            if nums[i] % 2 == 0:
                # Number is even, replace with 0
                nums[i] = 0
            else:
                # Number is odd, replace with 1
                nums[i] = 1
        
        # Step 3: Sort the modified array in non-decreasing order.
        # Since the array now only contains 0s and 1s, sorting will place all 0s before all 1s.
        nums.sort()
        
        # Return the resulting array after all operations.
        return nums