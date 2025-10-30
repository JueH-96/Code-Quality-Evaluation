from typing import List

class Solution:
    def resultArray(self, nums: List[int]) -> List[int]:
        # Initialize the two arrays with the first two elements as per the rules
        arr1, arr2 = [nums[0]], [nums[1]]

        # Process the remaining elements in nums
        for idx in range(2, len(nums)):
            if arr1[-1] > arr2[-1]:
                arr1.append(nums[idx])
            else:
                arr2.append(nums[idx])

        # Concatenate arr1 followed by arr2 to form the final result
        return arr1 + arr2