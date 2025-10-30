class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        n = len(grid)
        m = len(grid[0]) if n > 0 else 0
        R_count = [0] * n  # Number of ones in each row
        C_count = [0] * m  # Number of ones in each column

        # Count the number of ones in each row and column
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:
                    R_count[i] += 1
                    C_count[j] += 1

        result = 0
        # For each cell that contains a 1, calculate the number of right triangles
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:
                    # Exclude the current cell from the count
                    result += (R_count[i] - 1) * (C_count[j] - 1)
        return result