from typing import List

class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        n = len(grid)
        N = n * n # Total number of elements in the grid, and also the upper limit of the expected range (1 to N)
        
        seen_numbers = set()
        actual_sum = 0
        repeated_val = -1 
        
        # First pass: Iterate through the grid to find the repeated number and calculate the sum of all elements.
        for r in range(n):
            for c in range(n):
                val = grid[r][c]
                actual_sum += val
                
                # If the current value is already in our set of seen numbers, it's the repeated value.
                if val in seen_numbers:
                    repeated_val = val
                else:
                    # Otherwise, add it to the set.
                    seen_numbers.add(val)
        
        # Calculate the sum of integers from 1 to N.
        # The sum of an arithmetic series 1 + 2 + ... + N is N * (N + 1) / 2.
        expected_sum = N * (N + 1) // 2
        
        # Determine the missing number using the sum property.
        # The actual_sum of the grid elements differs from the expected_sum
        # because a missing number 'b' reduces the sum, and a repeated number 'a' increases it.
        # So, actual_sum = (expected_sum - missing_val) + repeated_val
        # Rearranging the formula to find missing_val:
        # missing_val = repeated_val - (actual_sum - expected_sum)
        missing_val = repeated_val - (actual_sum - expected_sum)
        
        return [repeated_val, missing_val]