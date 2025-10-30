from typing import List

class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        for d in range(-(n - 1), n):
            if d >= 0:
                start_row, start_col = d, 0
            else:
                start_row, start_col = 0, -d  # since d is negative
            
            # Collect elements in the diagonal
            elements = []
            r, c = start_row, start_col
            while r < n and c < n:
                elements.append(grid[r][c])
                r += 1
                c += 1
            
            # Sort the elements based on the diagonal's group
            if d >= 0:
                elements_sorted = sorted(elements, reverse=True)
            else:
                elements_sorted = sorted(elements)
            
            # Place the sorted elements back into the grid
            r, c = start_row, start_col
            idx = 0
            while r < n and c < n:
                grid[r][c] = elements_sorted[idx]
                r += 1
                c += 1
                idx += 1
        
        return grid