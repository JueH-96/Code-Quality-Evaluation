from typing import List

class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        # Group elements by their diagonal identifier d = i - j
        diags = {}
        for i in range(n):
            for j in range(n):
                d = i - j
                diags.setdefault(d, []).append(grid[i][j])
        
        # Sort each diagonal: non‐increasing for d >= 0, non‐decreasing for d < 0
        for d, lst in diags.items():
            if d >= 0:
                lst.sort(reverse=True)
            else:
                lst.sort()
        
        # Write the sorted values back into grid in the original traversal order
        ptrs = {d: 0 for d in diags}
        for i in range(n):
            for j in range(n):
                d = i - j
                grid[i][j] = diags[d][ptrs[d]]
                ptrs[d] += 1
        
        return grid