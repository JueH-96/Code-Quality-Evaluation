from typing import List

class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        # A single element or empty array is already "sorted"
        if n <= 1:
            return 0
        
        # Count the number of "drops" where nums[i] > nums[i+1] (circularly)
        drops = 0
        drop_index = -1
        for i in range(n):
            if nums[i] > nums[(i + 1) % n]:
                drops += 1
                drop_index = i
                if drops > 1:
                    return -1
        
        # If there are no drops, array is already sorted
        if drops == 0:
            return 0
        
        # Exactly one drop at drop_index
        # We need to right-shift so that the element at drop_index+1 comes to index 0
        # After k right shifts, index (drop_index+1 + k) % n == 0
        # => k == (n - (drop_index+1)) % n
        k = (n - (drop_index + 1)) % n
        return k