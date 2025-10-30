from typing import List

class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        total_sum = n * (n * n + 1) // 2
        actual_sum = sum(sum(row) for row in grid)
        set_sum = sum(set(sum(row) for row in grid))

        missing = (total_sum - set_sum)
        repeated = (actual_sum - set_sum)

        return [repeated, missing]