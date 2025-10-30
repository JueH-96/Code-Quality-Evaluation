from typing import List

class Solution:
    def dominantIndex(self, nums: List[int]) -> int:
        max_num = max(nums)
        max_index = nums.index(max_num)
        for i, num in enumerate(nums):
            if i != max_index and num * 2 > max_num:
                return -1
        return max_index

    def minimumIndex(self, nums: List[int]) -> int:
        dominant = self.dominantIndex(nums)
        for i in range(len(nums) - 1):
            left = nums[:i + 1]
            right = nums[i + 1:]
            if self.dominantIndex(left) == dominant and self.dominantIndex(right) == dominant:
                return i
        return -1