from typing import List

class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        max_val = n * n
        
        seen = [False] * (max_val + 1)
        repeating = -1
        total_sum = 0
        
        # Scan grid to find the repeating number and sum all entries
        for row in grid:
            for val in row:
                total_sum += val
                if seen[val]:
                    repeating = val
                else:
                    seen[val] = True
        
        # Compute expected sum of 1..n^2
        expected_sum = max_val * (max_val + 1) // 2
        
        # The missing number is what makes up the difference
        # expected_sum = (total_sum - repeating) + missing
        # => missing = expected_sum - (total_sum - repeating)
        missing = expected_sum - (total_sum - repeating)
        
        return [repeating, missing]