from typing import List

class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        left, right = [float('inf')] * rows, [float('-inf')] * rows
        top, bottom = [float('inf')] * cols, [float('-inf')] * cols
        ones = set()
        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    ones.add((i, j))
                    left[i] = min(left[i], j)
                    right[i] = max(right[i], j)
                    top[j] = min(top[j], i)
                    bottom[j] = max(bottom[j], i)
        
        min_area = float('inf')
        for (i, j) in ones:
            min_area = min(min_area, (right[i] - left[i] + 1) * (bottom[j] - top[j] + 1))
        
        return min_area