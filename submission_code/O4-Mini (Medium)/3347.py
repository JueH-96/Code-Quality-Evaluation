from typing import List

class Solution:
    def resultArray(self, nums: List[int]) -> List[int]:
        # Initialize arr1 with the first element and arr2 with the second
        arr1 = [nums[0]]
        arr2 = [nums[1]]
        
        # Distribute the remaining elements based on the last elements of arr1 and arr2
        for i in range(2, len(nums)):
            if arr1[-1] > arr2[-1]:
                arr1.append(nums[i])
            else:
                arr2.append(nums[i])
        
        # Concatenate arr1 and arr2 to form the result
        return arr1 + arr2