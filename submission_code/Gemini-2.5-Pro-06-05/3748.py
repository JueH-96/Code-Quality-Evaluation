import collections
from typing import List

class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        
        # A dictionary to store elements for each diagonal.
        # Diagonals are identified by the difference of their indices (i - j).
        diagonals = collections.defaultdict(list)

        # 1. Extract elements from the grid and group them by their diagonal.
        for i in range(n):
            for j in range(n):
                diagonals[i - j].append(grid[i][j])

        # 2. Sort each diagonal according to the specified rules.
        for key in diagonals:
            if key >= 0:
                # Bottom-left triangle and main diagonal: sort in non-increasing order.
                diagonals[key].sort(reverse=True)
            else:
                # Top-right triangle: sort in non-decreasing order.
                diagonals[key].sort()

        # 3. Reconstruct the grid with the sorted diagonal elements.
        # A dictionary of pointers (indices) tracks the current position in each sorted list.
        # Since we iterate through the grid in the same order as extraction (row-by-row),
        # we can simply place the sorted elements back in sequence.
        pointers = collections.defaultdict(int)
        for i in range(n):
            for j in range(n):
                diag_key = i - j
                # Fetch the next sorted element for the current diagonal.
                grid[i][j] = diagonals[diag_key][pointers[diag_key]]
                # Advance the pointer for that diagonal.
                pointers[diag_key] += 1
        
        return grid