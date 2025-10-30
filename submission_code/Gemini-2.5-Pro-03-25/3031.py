from typing import List

class Solution:
    """
    Implements the solution to construct the product matrix.
    The product matrix p is defined such that p[i][j] is the product of all elements
    in the input grid except for grid[i][j], with the result taken modulo 12345.
    """
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        """
        Calculates the product matrix p where p[i][j] is the product of all elements
        in grid except grid[i][j], modulo 12345.

        This method uses a two-pass approach inspired by the "product of array except self" problem.
        The matrix elements are processed in a flattened row-major order.
        
        1. The first pass calculates prefix products. For each element `grid[i][j]`, it computes the
           product of all elements that appear *before* it in the row-major order, modulo 12345.
           This prefix product is stored temporarily in the result matrix `p[i][j]`.
        
        2. The second pass calculates suffix products. It iterates through the elements in reverse
           row-major order. For each element `grid[i][j]`, it computes the product of all elements
           that appear *after* it in the row-major order, modulo 12345. This suffix product is
           then multiplied with the previously stored prefix product in `p[i][j]` to get the final
           result for `p[i][j]`, modulo 12345.

        All intermediate product calculations are performed modulo 12345 to prevent overflow
        and ensure the final result is within the required modulo range.

        Args:
            grid: A 0-indexed 2D integer matrix of size n * m.
                  Constraints: 1 <= n, m; 2 <= n * m <= 10^5; 1 <= grid[i][j] <= 10^9.

        Returns:
            A 0-indexed 2D integer matrix p of size n * m, representing the product matrix modulo 12345.
        """
        n = len(grid)
        # Constraints guarantee n >= 1
        m = len(grid[0])
        # Constraints guarantee m >= 1
        
        MOD = 12345

        # Initialize the result matrix p with 1s.
        # This initialization serves as the base case for multiplications.
        # In the first pass, p[i][j] will be populated with prefix products.
        # In the second pass, these prefix products will be multiplied by suffix products.
        p = [[1] * m for _ in range(n)]

        # Pass 1: Calculate prefix products (iterating in row-major order)
        prefix = 1  # Stores the running product of elements encountered so far.
        for i in range(n):
            for j in range(m):
                # The value at p[i][j] should be the product of elements BEFORE grid[i][j].
                # So, we assign the current prefix product to p[i][j] *before* updating prefix.
                p[i][j] = prefix
                
                # Update the prefix product to include the current element grid[i][j].
                # Apply modulo at each multiplication step to keep the numbers manageable
                # and ensure correctness under modulo arithmetic.
                prefix = (prefix * grid[i][j]) % MOD

        # Pass 2: Calculate suffix products (iterating in reverse row-major order)
        # and combine them with the prefix products stored in p.
        suffix = 1 # Stores the running product of elements encountered so far (in reverse).
        for i in range(n - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                # p[i][j] currently holds the prefix product (product of elements before grid[i][j]).
                # We need to multiply it by the suffix product (product of elements after grid[i][j]).
                # The current 'suffix' variable holds the product of elements strictly after grid[i][j].
                p[i][j] = (p[i][j] * suffix) % MOD
                
                # Update the suffix product to include the current element grid[i][j]
                # for the next element processed in the reverse iteration.
                # Apply modulo at each multiplication step.
                suffix = (suffix * grid[i][j]) % MOD

        return p