from typing import List
from collections import Counter

class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        # Flatten the grid into a 1D list
        flattened = []
        for row in grid:
            flattened.extend(row)
        
        # Use a counter to find the duplicate number (a)
        counts = Counter(flattened)
        a = [k for k, v in counts.items() if v == 2][0]
        
        # Calculate the actual sum and the expected sum
        n = len(grid)
        actual_sum = sum(flattened)
        expected_sum = (n * n) * (n * n + 1) // 2
        
        # Compute the missing number (b)
        b = expected_sum - (actual_sum - a)
        
        return [a, b]