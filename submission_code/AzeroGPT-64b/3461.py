from typing import List

class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        
        # Find all the columns with at least one '1'
        columns_with_ones = set()
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    columns_with_ones.add(c)
        
        min_area = float('inf')
        
        # For each pair of columns, find the smallest covering rectangle
        for left in columns_with_ones:
            for right in columns_with_ones:
                if left >= right:
                    continue
                
                # For this column pair, find the topmost and bottommost '1'
                top = rows
                bottom = -1
                for r in range(rows):
                    if grid[r][left] == 1 or grid[r][right] == 1:
                        top = min(top, r)
                        bottom = max(bottom, r)
                
                # If no '1' found between these two columns, skip
                if top > bottom:
                    continue
                
                # Calculate the current rectangle's area
                current_area = (right - left + 1) * (bottom - top + 1)
                min_area = min(min_area, current_area)
        
        return min_area