from typing import List

class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        score = 0
        while any(row for row in nums):
            removed = [max(row) for row in nums if row]
            score += max(removed)
            for i in range(len(nums)):
                if nums[i]:
                    nums[i].remove(max(nums[i]))
        return score