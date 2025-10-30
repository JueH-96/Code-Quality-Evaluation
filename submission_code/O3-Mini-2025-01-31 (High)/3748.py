from typing import List

class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        # Group all diagonals by their key = j - i.
        # Diagonals with key <= 0 (i.e. bottom-left triangle including the main diagonal)
        # will be sorted in non-increasing order.
        # Diagonals with key > 0 (i.e. top-right triangle) will be sorted in non-decreasing order.
        diagonals = {}
        for i in range(n):
            for j in range(n):
                key = j - i
                if key not in diagonals:
                    diagonals[key] = []
                diagonals[key].append(grid[i][j])
        
        # Sort each diagonal based on the rule:
        # key > 0: sort ascending (non-decreasing)
        # key <= 0: sort descending (non-increasing)
        for key in diagonals:
            if key > 0:
                diagonals[key].sort()
            else:
                diagonals[key].sort(reverse=True)
        
        # Place the sorted values back to their positions.
        # Iterate in the natural order (increasing row index) so that
        # each diagonal's sorted order is distributed from the top-left
        # (or appropriate starting point) downwards.
        for i in range(n):
            for j in range(n):
                key = j - i
                grid[i][j] = diagonals[key].pop(0)
        
        return grid