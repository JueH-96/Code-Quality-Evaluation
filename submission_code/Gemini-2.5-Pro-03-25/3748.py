import collections 
from typing import List 

class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        """
        Sorts the diagonals of the input square matrix grid according to specific rules.

        The diagonals are defined by cells (r, c) where r - c is constant.
        - Diagonals in the bottom-left triangle (where r - c >= 0, including the main diagonal)
          are sorted in non-increasing (descending) order.
        - Diagonals in the top-right triangle (where r - c < 0)
          are sorted in non-decreasing (ascending) order.

        Args:
            grid: An n x n square matrix of integers.

        Returns:
            The modified grid with diagonals sorted as specified.
        """
        
        n = len(grid)
        # Constraints state n >= 1, so n=0 check is technically not needed, 
        # but included for robustness.
        if n == 0:
            return []

        # Use defaultdict to store elements of each diagonal.
        # The key `r - c` uniquely identifies each diagonal parallel to the main diagonal.
        # The value is a list containing the elements belonging to that diagonal.
        diagonals = collections.defaultdict(list)

        # Step 1: Extract elements for each diagonal
        # Iterate through each cell of the grid.
        for r in range(n):
            for c in range(n):
                # Calculate the diagonal key for the current cell (r, c).
                diagonal_key = r - c
                # Append the element grid[r][c] to the list associated with its diagonal key.
                diagonals[diagonal_key].append(grid[r][c])

        # Step 2: Sort each diagonal based on the problem rules
        # Iterate through all the collected diagonals using their keys.
        for k in diagonals:
            # Check if the diagonal belongs to the bottom-left triangle (including the main diagonal).
            # This corresponds to diagonal keys k = r - c >= 0.
            if k >= 0:  
                # Sort the elements of this diagonal in non-increasing (descending) order.
                diagonals[k].sort(reverse=True)
            # Otherwise, the diagonal belongs to the top-right triangle.
            # This corresponds to diagonal keys k = r - c < 0.
            else: 
                # Sort the elements of this diagonal in non-decreasing (ascending) order.
                diagonals[k].sort() 

        # Step 3: Place the sorted elements back into the grid
        # Use a dictionary to keep track of the current index within each sorted diagonal list.
        # This ensures that elements are placed back into the grid cells along their respective
        # diagonals in the correct sorted order.
        indices = collections.defaultdict(int)

        # Iterate through each cell of the grid again.
        for r in range(n):
            for c in range(n):
                # Identify the diagonal key for the current cell (r, c).
                diagonal_key = r - c
                # Get the index of the next element to place from this diagonal's sorted list.
                current_index = indices[diagonal_key]
                # Assign the element from the sorted diagonal list at the current index 
                # back to the grid cell (r, c). This modifies the grid in place.
                grid[r][c] = diagonals[diagonal_key][current_index]
                # Increment the index for this diagonal, so the next cell on the same diagonal
                # gets the next element from the sorted list.
                indices[diagonal_key] += 1
                
        # Step 4: Return the modified grid.
        # The input grid has been modified in place to contain the sorted diagonals.
        return grid