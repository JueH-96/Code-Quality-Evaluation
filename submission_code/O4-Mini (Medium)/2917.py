from typing import List

class Solution:
    def countPairs(self, nums: List[int], target: int) -> int:
        nums.sort()
        count = 0
        left, right = 0, len(nums) - 1
        while left < right:
            if nums[left] + nums[right] < target:
                # All pairs (left, left+1), ..., (left, right) are valid
                count += (right - left)
                left += 1
            else:
                right -= 1
        return count