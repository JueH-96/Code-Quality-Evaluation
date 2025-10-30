from typing import List
from collections import defaultdict

class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        if n == 0:
            return grid
        
        diag_dict = defaultdict(lambda: {'elements': [], 'coords': []})
        
        for i in range(n):
            for j in range(n):
                d = i - j
                diag_dict[d]['elements'].append(grid[i][j])
                diag_dict[d]['coords'].append((i, j))
        
        for d in diag_dict:
            elements = diag_dict[d]['elements']
            coords = diag_dict[d]['coords']
            
            if d >= 0:
                elements.sort(reverse=True)
            else:
                elements.sort()
            
            for k in range(len(elements)):
                i, j = coords[k]
                grid[i][j] = elements[k]
        
        return grid