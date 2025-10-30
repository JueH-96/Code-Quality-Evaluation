class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        # Calculate number of rows (R) and columns (C)
        R, C = len(grid), len(grid[0])
        
        # Count how many 1's are in each row and each column
        row_counts = [0] * R
        col_counts = [0] * C
        
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 1:
                    row_counts[r] += 1
                    col_counts[c] += 1
        
        # For each 1 in the grid, compute the number of right triangles
        # using that cell as the "corner" of the right angle
        total_right_triangles = 0
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 1:
                    # (row_counts[r] - 1) possible partners horizontally
                    # (col_counts[c] - 1) possible partners vertically
                    total_right_triangles += (row_counts[r] - 1) * (col_counts[c] - 1)
        
        return total_right_triangles