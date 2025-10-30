class Solution:
    def numberOfRightTriangles(self, grid: list[list[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        row_counts = [sum(row) for row in grid]
        col_counts = [sum(grid[r][c] for r in range(rows)) for c in range(cols)]
        
        right_triangles = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    # Each 1 in the grid can be the vertex of a right triangle
                    # The number of triangles with this vertex is the product of the number of 1s
                    # in the same row and the number of 1s in the same column, minus one to not count the vertex itself twice
                    right_triangles += (row_counts[r] - 1) * (col_counts[c] - 1)
        
        return right_triangles