from typing import List

class Solution:
    def countPairs(self, nums: List[int], target: int) -> int:
        nums_sorted = sorted(nums)
        count = 0
        left = 0
        right = len(nums_sorted) - 1
        while left < right:
            if nums_sorted[left] + nums_sorted[right] < target:
                count += right - left
                left += 1
            else:
                right -= 1
        return count