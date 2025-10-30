from collections import Counter
from typing import List

class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        total = n * n
        expected_sum = total * (total + 1) // 2
        
        # Calculate the sum of all elements in the grid
        sum_grid = sum(num for row in grid for num in row)
        D = sum_grid - expected_sum
        
        # Count the frequency of each number
        counts = Counter()
        for row in grid:
            counts.update(row)
        
        # Find the duplicate number (a)
        a = None
        for num in counts:
            if counts[num] == 2:
                a = num
                break
        
        # Calculate the missing number (b)
        b = a - D
        
        return [a, b]