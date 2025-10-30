from typing import List

class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        # count the number of "drops" (i.e., where an element is greater than the next)
        drops = 0
        drop_index = -1
        
        # Check consecutive pairs from 0 to n-2
        for i in range(n - 1):
            if nums[i] > nums[i + 1]:
                drops += 1
                drop_index = i
        
        # Check the circular condition: last element and first element.
        if nums[-1] > nums[0]:
            drops += 1
        
        if drops == 0:
            # Already sorted
            return 0
        elif drops == 1:
            # The number of right shifts needed is:
            # In a rotation, the smallest element will eventually land at index 0.
            # If the drop is at position drop_index, then the smallest element is at drop_index + 1.
            # We need n - (drop_index + 1) shifts.
            return n - drop_index - 1
        else:
            return -1