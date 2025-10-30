from typing import List

class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        frequency = {}
        n = len(grid)
        for row in grid:
            for num in row:
                frequency[num] = frequency.get(num, 0) + 1
        a = b = -1
        for num in range(1, n*n +1):
            if frequency.get(num, 0) == 2:
                a = num
            elif frequency.get(num,0) ==0:
                b = num
        return [a, b]