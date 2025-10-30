from typing import List

class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        count = [0] * (n * n + 1)
        
        for row in grid:
            for num in row:
                count[num] += 1
        
        repeated = 0
        missing = 0
        for i in range(1, n * n + 1):
            if count[i] == 2:
                repeated = i
            elif count[i] == 0:
                missing = i
        
        return [repeated, missing]