from collections import defaultdict
from typing import List

class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        
        # collect the values of each diagonal, key = row - col
        diags = defaultdict(list)
        for r in range(n):
            for c in range(n):
                diags[r - c].append(grid[r][c])
        
        # sort every diagonal according to the rule
        #   d >= 0  -> non-increasing  (descending)
        #   d < 0   -> non-decreasing (ascending)
        for d, arr in diags.items():
            if d >= 0:
                arr.sort(reverse=True)
            else:
                arr.sort()
        
        # indices to know the next element to place back for each diagonal
        idx = defaultdict(int)
        
        # write the values back to the grid in top-left → bottom-right order
        for r in range(n):
            for c in range(n):
                d = r - c
                grid[r][c] = diags[d][idx[d]]
                idx[d] += 1
        
        return grid