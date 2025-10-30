from typing import List

class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        
        # Dictionary to store elements for each diagonal.
        # Key: diagonal index k = i - j
        # Value: list of elements on that diagonal
        diagonals = {}
        
        # Dictionary to store coordinates for each diagonal,
        # preserving the original order for putting elements back.
        # Key: diagonal index k = i - j
        # Value: list of (row, col) tuples on that diagonal
        coords = {}

        # 1. Traverse the matrix and group elements by diagonal index (i - j)
        for i in range(n):
            for j in range(n):
                k = i - j
                if k not in diagonals:
                    diagonals[k] = []
                    coords[k] = []
                diagonals[k].append(grid[i][j])
                coords[k].append((i, j))

        # 2. Sort the elements for each diagonal based on its position
        # The diagonal index k ranges from -(n-1) to n-1
        for k in range(-(n - 1), n):
            if k in diagonals: # Check if this diagonal exists
                diag_values = diagonals[k]
                
                # Diagonals in the bottom-left triangle (including the main diagonal)
                # have i - j >= 0 (i.e., k >= 0). Sort non-increasingly.
                if k >= 0:
                    diag_values.sort(reverse=True)
                
                # Diagonals in the top-right triangle have i - j < 0 (i.e., k < 0).
                # Sort non-decreasingly.
                else:
                    diag_values.sort()

        # 3. Place the sorted elements back into the matrix
        # Iterate through the diagonal indices again
        for k in range(-(n - 1), n):
            if k in diagonals:
                diag_coords = coords[k]
                diag_values = diagonals[k] # This list is now sorted

                # Place the sorted values back into the grid using the stored coordinates
                # The order of coordinates in coords[k] matches the order
                # in which elements were extracted into diagonals[k].
                for idx, (r, c) in enumerate(diag_coords):
                    grid[r][c] = diag_values[idx]

        return grid