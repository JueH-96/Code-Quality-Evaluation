from typing import List

class Solution:
  def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
    MOD = 12345
    
    n = len(grid)
    # Constraints state 1 <= n, m <= 10^5, so n and m are at least 1.
    # Thus, grid[0] is safe to access.
    m = len(grid[0])

    # Initialize the product matrix p.
    # In the first pass, p[i][j] will store the product of elements "before" grid[i][j].
    # In the second pass, this will be multiplied by the product of elements "after" grid[i][j].
    p = [[0] * m for _ in range(n)]

    # First pass: Calculate prefix products
    # current_prefix_prod stores the product of all elements encountered *before*
    # the current grid[i][j] element, in row-major order.
    # All products are taken modulo MOD.
    current_prefix_prod = 1
    for i in range(n):
      for j in range(m):
        p[i][j] = current_prefix_prod
        current_prefix_prod = (current_prefix_prod * grid[i][j]) % MOD
    
    # Second pass: Calculate suffix products and combine
    # current_suffix_prod stores the product of all elements encountered *after*
    # the current grid[i][j] element (when iterating in reverse row-major order).
    # All products are taken modulo MOD.
    current_suffix_prod = 1
    for i in range(n - 1, -1, -1):
      for j in range(m - 1, -1, -1):
        # p[i][j] currently holds the prefix product.
        # Multiply it by the suffix product to get the final result for p[i][j].
        p[i][j] = (p[i][j] * current_suffix_prod) % MOD
        # Update current_suffix_prod for the next element (which is to the "left"
        # or "up" in this reverse iteration).
        current_suffix_prod = (current_suffix_prod * grid[i][j]) % MOD
        
    return p