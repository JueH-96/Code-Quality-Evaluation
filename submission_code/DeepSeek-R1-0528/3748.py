from collections import defaultdict
from typing import List

class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        diagonals = defaultdict(list)
        
        for i in range(n):
            for j in range(n):
                k = i - j
                diagonals[k].append(grid[i][j])
        
        sorted_diagonals = {}
        for k, vals in diagonals.items():
            if k >= 0:
                sorted_diagonals[k] = sorted(vals, reverse=True)
            else:
                sorted_diagonals[k] = sorted(vals)
        
        iters = {k: iter(v) for k, v in sorted_diagonals.items()}
        
        res = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                k = i - j
                res[i][j] = next(iters[k])
        
        return res