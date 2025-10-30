from typing import List

class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        score = 0
        while any(nums):
            max_values = []
            for row in nums:
                max_value = max(row)
                row.remove(max_value)
                max_values.append(max_value)
            score += max(max_values)
        return score