from typing import List

class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        flat_grid = [num for row in grid for num in row]
        k = len(flat_grid)
        prefix = [1] * k
        for i in range(k):
            if i > 0:
                prefix[i] = (prefix[i - 1] * flat_grid[i - 1]) % 12345
            else:
                prefix[i] = 1
        suffix = [1] * k
        for i in range(k - 1, -1, -1):
            if i < k - 1:
                suffix[i] = (suffix[i + 1] * flat_grid[i + 1]) % 12345
            else:
                suffix[i] = 1
        p_flat = [(prefix[i] * suffix[i]) % 12345 for i in range(k)]
        n = len(grid)
        m = len(grid[0]) if n > 0 else 0
        p = []
        idx = 0
        for row in grid:
            row_length = len(row)
            p.append(p_flat[idx:idx + row_length])
            idx += row_length
        return p