from typing import List

class Solution:
    def resultArray(self, nums: List[int]) -> List[int]:
        """
        Distributes elements of `nums` into two arrays, `arr1` and `arr2`,
        based on a specific set of rules, and returns their concatenation.
        """
        
        # Per constraints, len(nums) >= 3, so nums[0] and nums[1] are always valid.
        
        # Initialize arr1 and arr2 based on the first two operations.
        # Operation 1: nums[0] goes to arr1.
        arr1 = [nums[0]]
        # Operation 2: nums[1] goes to arr2.
        arr2 = [nums[1]]
        
        # Process the remaining elements from the third element (index 2) onwards.
        for i in range(2, len(nums)):
            # Compare the last elements of the two arrays to decide where to append the current element.
            if arr1[-1] > arr2[-1]:
                # If arr1's last element is greater, append nums[i] to arr1.
                arr1.append(nums[i])
            else:
                # Otherwise, append nums[i] to arr2.
                arr2.append(nums[i])
        
        # The final result is the concatenation of arr1 and arr2.
        return arr1 + arr2