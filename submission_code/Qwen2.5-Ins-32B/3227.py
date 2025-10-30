from typing import List

class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        flat_grid = [num for row in grid for num in row]
        num_count = [0] * (n * n + 1)
        
        for num in flat_grid:
            num_count[num] += 1
        
        repeated = missing = None
        for i in range(1, n * n + 1):
            if num_count[i] == 2:
                repeated = i
            elif num_count[i] == 0:
                missing = i
        
        return [repeated, missing]