from typing import List

class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        nums = [0] * (n * n + 1)
        for row in grid:
            for num in row:
                nums[num] += 1
        a, b = 0, 0
        for i in range(1, n * n + 1):
            if nums[i] == 2:
                a = i
            if nums[i] == 0:
                b = i
        return [a, b]