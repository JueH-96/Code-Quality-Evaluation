class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        
        # Initialize boundaries
        left, right = cols, 0
        top, bottom = rows, 0
        
        # Find the boundaries of the rectangle
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    left = min(left, c)
                    right = max(right, c)
                    top = min(top, r)
                    bottom = max(bottom, r)
        
        # If no 1's found, return 0 (though the problem guarantees at least one 1)
        if left > right or top > bottom:
            return 0
        
        # Calculate area
        area = (right - left + 1) * (bottom - top + 1)
        return area