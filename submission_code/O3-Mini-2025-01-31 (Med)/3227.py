from typing import List

class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        total = n * n
        # Create a frequency array for numbers 1 to total
        freq = [0] * (total + 1)
        
        # Count the occurrences of each number in the grid
        for row in grid:
            for num in row:
                freq[num] += 1
        
        repeated, missing = -1, -1
        # Check for the number that is repeated (frequency 2) and missing (frequency 0)
        for num in range(1, total + 1):
            if freq[num] == 2:
                repeated = num
            elif freq[num] == 0:
                missing = num
                
        return [repeated, missing]