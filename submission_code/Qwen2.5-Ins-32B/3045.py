from typing import List

class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0
        
        # Find the index where the array is not in increasing order
        shift_index = -1
        for i in range(1, n):
            if nums[i] < nums[i - 1]:
                if shift_index == -1:
                    shift_index = i
                else:
                    # If there are more than one such indices, return -1
                    return -1
        
        # If the array is already sorted
        if shift_index == -1:
            return 0
        
        # Check if the array can be sorted with right shifts
        if nums[0] < nums[-1]:
            return -1
        
        # Calculate the number of right shifts required
        return n - shift_index