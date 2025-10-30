from typing import List

class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        # Find the position of the smallest element (which should become index 0 after shifting)
        pos = nums.index(min(nums))
        # Compute how many right shifts are needed to bring index pos to index 0
        k = (n - pos) % n
        
        # Perform the rotation by k
        if k:
            rotated = nums[-k:] + nums[:-k]
        else:
            rotated = nums[:]
        
        # Check if the rotated array is sorted
        if rotated == sorted(nums):
            return k
        return -1