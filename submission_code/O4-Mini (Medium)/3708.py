from typing import List

class Solution:
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        result = []
        step = 0
        n = len(grid)
        m = len(grid[0]) if n > 0 else 0
        
        for i in range(n):
            # Determine direction: left-to-right if even row, else right-to-left
            if i % 2 == 0:
                cols = range(m)
            else:
                cols = range(m - 1, -1, -1)
            
            for j in cols:
                # On every alternate step (0-based even), take the value
                if step % 2 == 0:
                    result.append(grid[i][j])
                step += 1
        
        return result