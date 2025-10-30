from typing import List

class Solution:
    def resultArray(self, nums: List[int]) -> List[int]:
        # Initialize the two arrays
        arr1 = []
        arr2 = []

        # First operation: append nums[0] (which is nums[1] in 1-indexed problem statement) to arr1
        arr1.append(nums[0])

        # Second operation: append nums[1] (which is nums[2] in 1-indexed problem statement) to arr2
        arr2.append(nums[1])

        # Subsequent operations for the remaining elements
        # The loop starts from the third element (index 2 in 0-indexed list)
        for i in range(2, len(nums)):
            # Get the last elements of arr1 and arr2 for comparison
            last_arr1 = arr1[-1]
            last_arr2 = arr2[-1]

            # Apply the distribution rule
            if last_arr1 > last_arr2:
                arr1.append(nums[i])
            else:
                arr2.append(nums[i])

        # Concatenate arr1 and arr2 to form the final result array
        return arr1 + arr2