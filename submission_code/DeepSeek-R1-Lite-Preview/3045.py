from typing import List

class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0  # A single element is already sorted
        
        drop_count = 0
        drop_index = -1
        
        for i in range(n):
            if nums[i] > nums[(i + 1) % n]:
                drop_count += 1
                drop_index = i
                if drop_count > 1:
                    return -1  # More than one drop, cannot sort with right shifts
        
        if drop_count == 0:
            return 0  # Already sorted
        elif drop_count == 1:
            # Check if the array can be rotated to sorted order
            if nums[-1] <= nums[0]:
                # Number of right shifts is n - (drop_index + 1)
                return n - (drop_index + 1)
            else:
                return -1
        else:
            return -1  # More than one drop, cannot sort