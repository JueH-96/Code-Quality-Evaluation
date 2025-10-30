from typing import List

class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        
        # Constraints: 1 <= n <= 10. So n is at least 1.
        # The grid is guaranteed to be n x n.

        # Iterate over all diagonals. A diagonal is defined by a constant value of r - c.
        # Let D_val = r - c.
        # Minimum D_val: for cell (0, n-1), D_val = 0 - (n-1) = 1-n.
        # Maximum D_val: for cell (n-1, 0), D_val = (n-1) - 0 = n-1.
        # So, D_val ranges from 1-n to n-1, inclusive.
        # In Python's range(start, stop), 'stop' is exclusive, so range(1-n, n).
        for D_val in range(1 - n, n): 
            diag_elements = []  # To store elements of the current diagonal
            coordinates = []    # To store (row, col) of these elements for re-populating
            
            # Determine the starting (row_start, col_start) for the current diagonal D_val.
            # Elements on a diagonal are processed from top-left to bottom-right.
            if D_val >= 0: 
                # This diagonal is in the bottom-left triangle (where r >= c)
                # or it's the main diagonal (where r == c).
                # For r - c = D_val:
                # The first element on this diagonal (top-most, left-most) occurs
                # when c is minimal (0) or r is minimal (D_val).
                # So, (row_start, col_start) = (D_val, 0).
                row_start, col_start = D_val, 0
            else: # D_val < 0
                # This diagonal is in the top-right triangle (where r < c).
                # For r - c = D_val (or c - r = -D_val):
                # The first element on this diagonal (top-most, left-most) occurs
                # when r is minimal (0) or c is minimal (-D_val).
                # So, (row_start, col_start) = (0, -D_val).
                row_start, col_start = 0, -D_val
            
            # Current position for iterating along the diagonal
            curr_r, curr_c = row_start, col_start
            
            # Extract all elements and their coordinates from the current diagonal.
            while curr_r < n and curr_c < n: # Stay within grid boundaries
                diag_elements.append(grid[curr_r][curr_c])
                coordinates.append((curr_r, curr_c))
                curr_r += 1 # Move to the next element on the diagonal
                curr_c += 1 # (which is one step down and one step right)
            
            # Sort the extracted elements according to the problem's rules.
            if D_val >= 0: 
                # Diagonals in bottom-left triangle (including the main diagonal)
                # are sorted in non-increasing (descending) order.
                diag_elements.sort(reverse=True) 
            else: 
                # Diagonals in top-right triangle
                # are sorted in non-decreasing (ascending) order.
                diag_elements.sort() # Default sort is ascending (reverse=False)
            
            # Place the sorted elements back into the grid at their original positions.
            for i in range(len(diag_elements)):
                row_idx, col_idx = coordinates[i]
                grid[row_idx][col_idx] = diag_elements[i]
                
        return grid