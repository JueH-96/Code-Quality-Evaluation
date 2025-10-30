from typing import List

class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        total_nums = n * n
        expected_sum = total_nums * (total_nums + 1) // 2
        expected_sum_sq = total_nums * (total_nums + 1) * (2 * total_nums + 1) // 6
        actual_sum = sum(num for row in grid for num in row)
        actual_sum_sq = sum(num * num for row in grid for num in row)
        diff_sum = actual_sum - expected_sum  # A - B
        diff_sum_sq = actual_sum_sq - expected_sum_sq  # A^2 - B^2
        sum_ab = diff_sum_sq // diff_sum  # A + B
        repeated = (sum_ab + diff_sum) // 2
        missing = (sum_ab - diff_sum) // 2
        return [repeated, missing]