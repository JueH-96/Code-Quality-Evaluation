from typing import List

class Solution:
    def resultArray(self, nums: List[int]) -> List[int]:
        # initialize the two arrays with the first two elements
        arr1 = [nums[0]]
        arr2 = [nums[1]]
        
        # distribute the remaining elements
        for x in nums[2:]:
            if arr1[-1] > arr2[-1]:
                arr1.append(x)
            else:
                arr2.append(x)
        
        # concatenate and return
        return arr1 + arr2