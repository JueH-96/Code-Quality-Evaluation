from typing import List

class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        # There are numbers from 1 to n*n
        max_val = n * n
        counts = [0] * (max_val + 1)
        
        # Count occurrences of each number in the grid
        for row in grid:
            for val in row:
                counts[val] += 1
        
        # Identify the repeated (count == 2) and missing (count == 0) values
        repeated = missing = -1
        for num in range(1, max_val + 1):
            if counts[num] == 2:
                repeated = num
            elif counts[num] == 0:
                missing = num
        
        return [repeated, missing]