from typing import List

class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        max_val = n * n
        # Frequency list to count occurrences for numbers 1 to n*n.
        freq = [0] * (max_val + 1)
        
        # Count occurrences of each value in the grid.
        for row in grid:
            for num in row:
                freq[num] += 1
        
        repeated = -1
        missing = -1
        
        # Determine the repeated and missing values.
        for i in range(1, max_val + 1):
            if freq[i] == 0:
                missing = i
            elif freq[i] == 2:
                repeated = i
        
        return [repeated, missing]