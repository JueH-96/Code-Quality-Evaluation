import collections
from typing import List

class Solution:
    """
    Finds the repeated and missing number in a grid where numbers 
    from 1 to n*n should appear exactly once, but one number 'a' is repeated 
    and one number 'b' is missing.
    """
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        """
        Args:
            grid: A n x n 2D list of integers, where n is the number of rows/columns.
                  The values are expected to be in the range [1, n*n].

        Returns:
            A list containing two integers: [repeated_number, missing_number].
        """
        n = len(grid)
        # Calculate the maximum value expected in the grid (n*n)
        max_val = n * n
        
        # Use a frequency array (list) to store counts of numbers from 1 to n*n.
        # Initialize a list of size max_val + 1 with zeros.
        # Index 0 will be unused, indices 1 to max_val will correspond to the numbers.
        counts = [0] * (max_val + 1) 
        
        # Iterate through each cell of the grid
        for r in range(n):
            for c in range(n):
                # Get the value from the current cell
                val = grid[r][c]
                # Increment the count for this value in the frequency array
                # The problem constraints guarantee val is within [1, n*n]
                counts[val] += 1

        repeated_num = -1  # Placeholder for the repeated number
        missing_num = -1   # Placeholder for the missing number

        # Iterate through the expected range of numbers [1, n*n]
        # Check the frequency count for each number.
        for i in range(1, max_val + 1):
            if counts[i] == 2:
                # If a number appears twice, it's the repeated number 'a'
                repeated_num = i
            elif counts[i] == 0:
                # If a number has a count of 0, it's the missing number 'b'
                missing_num = i
        
        # The problem constraints guarantee that exactly one number is repeated
        # and exactly one number is missing, so both repeated_num and missing_num
        # will be assigned a value other than -1 by the end of the loop.
        
        # Return the result as a list [repeated_number, missing_number]
        return [repeated_num, missing_num]