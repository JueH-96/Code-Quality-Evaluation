class Solution:
    def numberOfRightTriangles(self, grid: List[List[int]]) -> int:
        # Precompute row sums and column sums
        N = len(grid)
        M = len(grid[0]) if N > 0 else 0
        
        row_ones = [0] * N
        for i in range(N):
            row_ones[i] = sum(grid[i])
        
        col_ones = [0] * M
        for j in range(M):
            col_ones[j] = sum(grid[i][j] for i in range(N))
        
        total_triangles = 0
        for i in range(N):
            for j in range(M):
                if grid[i][j] == 1:
                    triangles_at_ij = (row_ones[i] - 1) * (col_ones[j] - 1)
                    total_triangles += triangles_at_ij
        return total_triangles