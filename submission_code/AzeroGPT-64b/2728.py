import numpy as np
class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        score = 0
        for i in range(len(nums)):
            nums[i] = sorted(nums[i])
        temp = list(zip(*nums))
        for i in temp:
            score += max(i)
        return score