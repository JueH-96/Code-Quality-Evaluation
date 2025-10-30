class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        n = len(grid)
        m = len(grid[0])
        
        # Initialize boundaries with extreme values
        top = n
        bottom = -1
        left = m
        right = -1
        
        # Find the boundaries of the rectangle that covers all 1's
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:
                    top = min(top, i)
                    bottom = max(bottom, i)
                    left = min(left, j)
                    right = max(right, j)
        
        # The area of the rectangle is given by the dimensions height x width
        area = (bottom - top + 1) * (right - left + 1)
        return area