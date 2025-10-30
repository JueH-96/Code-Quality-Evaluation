from typing import List

class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        m = len(grid[0])
        MOD = 12345
        N = n * m # Total number of elements in the grid

        # Calculate prefix products in the flattened sequence
        # We can treat the 2D grid elements as a 1D sequence in row-major order.
        # prefix_prod[k] stores the product of elements from the start of the flattened sequence
        # up to the element at flattened index k-1.
        prefix_prod = [1] * N
        current_prod = 1
        k = 0 # Flattened index
        for i in range(n):
            for j in range(m):
                # Store the product of elements *before* the current element grid[i][j] (flattened index k)
                prefix_prod[k] = current_prod
                # Update the current product to include the current element
                current_prod = (current_prod * grid[i][j]) % MOD
                k += 1

        # Calculate suffix products in the flattened sequence
        # suffix_prod[k] stores the product of elements from flattened index k+1 to the end of the sequence.
        suffix_prod = [1] * N
        current_prod = 1
        k = N - 1 # Start from the last flattened index
        # Iterate backward through the grid indices (which corresponds to backward through flattened indices)
        for i in range(n - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                # Store the product of elements *after* the current element grid[i][j] (flattened index k)
                suffix_prod[k] = current_prod
                # Update the current product to include the current element
                current_prod = (current_prod * grid[i][j]) % MOD
                k -= 1

        # Construct the result matrix
        result_matrix = [[0] * m for _ in range(n)]
        k = 0 # Reset flattened index
        for i in range(n):
            for j in range(m):
                # The element grid[i][j] corresponds to the element at flattened index k = i * m + j.
                # The product of all elements EXCEPT grid[i][j] is the product of elements
                # before flattened index k (prefix_prod[k]) multiplied by the product of elements
                # after flattened index k (suffix_prod[k]).
                result_matrix[i][j] = (prefix_prod[k] * suffix_prod[k]) % MOD
                k += 1

        return result_matrix