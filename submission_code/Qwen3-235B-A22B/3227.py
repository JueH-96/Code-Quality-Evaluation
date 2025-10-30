from typing import List

class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        n_squared = n * n
        
        # Calculate the expected sum and sum of squares of numbers from 1 to n_squared
        expected_sum = n_squared * (n_squared + 1) // 2
        expected_sq = n_squared * (n_squared + 1) * (2 * n_squared + 1) // 6
        
        # Calculate the actual sum and sum of squares from the grid
        sum_grid = 0
        sum_sq_grid = 0
        for row in grid:
            for num in row:
                sum_grid += num
                sum_sq_grid += num * num
        
        # Compute the differences
        sum_diff = expected_sum - sum_grid  # equals (b - a)
        sum_sq_diff = expected_sq - sum_sq_grid  # equals (b^2 - a^2)
        
        # Calculate sum_plus which is (a + b)
        sum_plus = sum_sq_diff // sum_diff
        
        # Solve for a and b
        a = (sum_plus - sum_diff) // 2
        b = (sum_plus + sum_diff) // 2
        
        return [a, b]