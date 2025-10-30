from typing import List

class Solution:
    def resultArray(self, nums: List[int]) -> List[int]:
        if len(nums) < 2:
            return nums  # Though per constraints, this might not be necessary
        
        arr1 = [nums[0]]
        arr2 = [nums[1]]
        
        for num in nums[2:]:
            if arr1[-1] > arr2[-1]:
                arr1.append(num)
            else:
                arr2.append(num)
        
        return arr1 + arr2