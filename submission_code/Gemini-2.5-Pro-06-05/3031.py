from typing import List

class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        """
        Calculates the product matrix p where p[i][j] is the product of all elements
        in grid except grid[i][j], with all calculations done modulo 12345.

        The solution uses a two-pass approach analogous to the "Product of Array
        Except Self" problem, treating the 2D grid as a flattened 1D array.
        This avoids division and is efficient in terms of time and space.

        Time Complexity: O(n * m), where n is the number of rows and m is the
                         number of columns, as we iterate through the grid twice.
        Space Complexity: O(n * m) to store the result matrix p. The auxiliary
                          space complexity is O(1).
        """
        MOD = 12345

        # Get the dimensions of the grid.
        # Constraints ensure n and m are at least 1.
        n = len(grid)
        m = len(grid[0])

        # Initialize the result matrix `p`. The initial values do not matter
        # as they will be overwritten.
        p = [[0] * m for _ in range(n)]

        # --- Pass 1: Forward traversal to calculate prefix products ---
        # `prefix_product` stores the running product of elements encountered so far
        # when traversing the grid in row-major order.
        prefix_product = 1
        for i in range(n):
            for j in range(m):
                # For the cell (i, j), the product of all preceding elements is
                # the current value of `prefix_product`.
                p[i][j] = prefix_product
                
                # Update the prefix product to include the current grid element.
                prefix_product = (prefix_product * grid[i][j]) % MOD

        # --- Pass 2: Backward traversal to multiply by suffix products ---
        # `suffix_product` stores the running product of elements from the end.
        suffix_product = 1
        for i in range(n - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                # Multiply the existing prefix product in p[i][j] by the
                # suffix product of all elements that follow grid[i][j].
                p[i][j] = (p[i][j] * suffix_product) % MOD
                
                # Update the suffix product to include the current grid element
                # for the next iteration in the backward pass.
                suffix_product = (suffix_product * grid[i][j]) % MOD
                
        return p