from typing import List

class Solution:
    def resultArray(self, nums: List[int]) -> List[int]:
        # Initialize two empty lists to store the elements
        arr1 = [nums[0]]
        arr2 = [nums[1]]
        
        # Iterate over the rest of the elements in the input list
        for i in range(2, len(nums)):
            # If the last element of arr1 is greater than the last element of arr2, append nums[i] to arr1
            if arr1[-1] > arr2[-1]:
                arr1.append(nums[i])
            # Otherwise, append nums[i] to arr2
            else:
                arr2.append(nums[i])
        
        # Return the result by concatenating arr1 and arr2
        return arr1 + arr2