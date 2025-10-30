class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        
        # Precompute row sums and column sums
        row_sums = [sum(grid[i]) for i in range(m)]
        col_sums = [sum(grid[i][j] for i in range(m)) for j in range(n)]
        
        result = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    # Count of 1's in same row (excluding current cell)
                    row_count = row_sums[i] - 1
                    # Count of 1's in same column (excluding current cell)
                    col_count = col_sums[j] - 1
                    # Add to result
                    result += row_count * col_count
        
        return result