from typing import List
from collections import defaultdict

class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        if n == 0:
            return []

        # Dictionary to store elements for each diagonal, identified by i - j
        # The keys will be the difference (i - j), and values will be lists of elements on that diagonal
        diagonals = defaultdict(list)

        # 1. Extract elements for each diagonal
        # Iterate through the grid and group elements by their diagonal identifier (i - j)
        for i in range(n):
            for j in range(n):
                diff = i - j
                diagonals[diff].append(grid[i][j])

        # 2. Sort each diagonal based on the rules
        # Iterate through the collected diagonal lists
        for diff in diagonals:
            if diff < 0: # Diagonals in the top-right triangle (where i < j, so i - j < 0)
                # Sort in non-decreasing (ascending) order
                diagonals[diff].sort()
            else: # Diagonals in the bottom-left triangle (where i >= j, so i - j >= 0), including the main diagonal
                # Sort in non-increasing (descending) order
                diagonals[diff].sort(reverse=True)

        # 3. Put sorted elements back into the grid
        # We need a way to track the current position (index) in the sorted list for each diagonal
        diag_pointers = defaultdict(int)

        # Create a new grid to store the result. Modifying in-place is also an option.
        result_grid = [[0] * n for _ in range(n)]

        # Iterate through the grid cells again to place the sorted elements back
        for i in range(n):
            for j in range(n):
                diff = i - j
                # Get the next sorted element from the corresponding diagonal list using the pointer
                sorted_element = diagonals[diff][diag_pointers[diff]]
                # Place the element in the result grid at the original (i, j) position
                result_grid[i][j] = sorted_element
                # Move the pointer for this diagonal to the next element for the next cell on this diagonal
                diag_pointers[diff] += 1

        return result_grid