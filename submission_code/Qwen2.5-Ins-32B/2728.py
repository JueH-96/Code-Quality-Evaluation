from typing import List

class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        score = 0
        while any(nums):
            max_in_row = []
            for row in nums:
                if row:
                    max_val = max(row)
                    max_in_row.append(max_val)
                    row.remove(max_val)
            score += max(max_in_row)
        return score