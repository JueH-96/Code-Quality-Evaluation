class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)

        # Iterate through all possible diagonal differences d = i - j
        # The range of d is from -(n-1) to n-1.
        # These differences correspond to diagonals:
        # d = n-1: bottom-left corner element (n-1, 0)
        # d = 0: main diagonal
        # d = -(n-1): top-right corner element (0, n-1)
        for d in range(-(n - 1), n):
            # Collect elements for the current diagonal
            diag_elements = []
            
            # Determine the range of row indices i for the current diagonal.
            # A point (i, j) is on the diagonal with difference d if i - j = d, which means j = i - d.
            # The point (i, j) must be within the grid bounds: 0 <= i < n and 0 <= j < n.
            # Substituting j: 0 <= i - d < n  => d <= i < n + d.
            # Combining with 0 <= i < n: the valid range for i is max(0, d) <= i < min(n, n + d).
            row_start = max(0, d)
            row_end = min(n, n + d)

            # Iterate through the valid row indices for this diagonal
            for i in range(row_start, row_end):
                j = i - d # Calculate the corresponding column index
                diag_elements.append(grid[i][j])

            # Sort the diagonal elements based on the rule
            # Diagonals in the bottom-left triangle (including main diagonal, i.e., i - j >= 0 or d >= 0)
            # should be sorted in non-increasing order (descending).
            # Diagonals in the top-right triangle (i.e., i - j < 0 or d < 0)
            # should be sorted in non-decreasing order (ascending).
            if d >= 0:
                # Sort in descending order for bottom-left and main diagonals
                diag_elements.sort(reverse=True)
            else: # d < 0
                # Sort in ascending order for top-right diagonals
                diag_elements.sort()

            # Place the sorted elements back into the grid
            k = 0 # index for the sorted diag_elements list
            # Iterate through the valid row indices again to place the sorted elements
            for i in range(row_start, row_end):
                j = i - d # Calculate the corresponding column index
                grid[i][j] = diag_elements[k]
                k += 1

        # Return the modified grid
        return grid