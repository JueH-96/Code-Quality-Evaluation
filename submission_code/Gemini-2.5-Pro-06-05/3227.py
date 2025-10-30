class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        """
        Finds the repeating and missing numbers in a n*n grid.

        The grid is supposed to contain each number from 1 to n*n exactly once,
        but one number 'a' is repeated and one number 'b' is missing.

        Args:
            grid: A 0-indexed 2D integer matrix of size n*n.

        Returns:
            A list [a, b] where 'a' is the repeated number and 'b' is the missing number.
        """
        n = len(grid)
        # The total number of elements, and the upper bound of the number range.
        N = n * n
        
        # We use a frequency array for numbers 1 to N.
        # The size is N+1 to allow for 1-based indexing.
        counts = [0] * (N + 1)
        
        grid_sum = 0
        repeated_num = -1

        # A single pass through the grid to calculate the grid's sum and
        # find the repeated number using the frequency counter.
        for r in range(n):
            for c in range(n):
                num = grid[r][c]
                
                grid_sum += num
                
                counts[num] += 1
                # The first number to have a count of 2 is the repeated number.
                # The problem guarantees exactly one number is repeated twice.
                if counts[num] == 2:
                    repeated_num = num

        # The expected sum of a perfect sequence of numbers from 1 to N.
        expected_sum = N * (N + 1) // 2
        
        # Let 'a' be the repeated number and 'b' be the missing number.
        # The actual sum of the grid is related to the expected sum by:
        # grid_sum = expected_sum - b + a
        # We rearrange this to solve for the missing number 'b':
        # b = expected_sum - grid_sum + a
        missing_num = expected_sum - grid_sum + repeated_num
        
        return [repeated_num, missing_num]