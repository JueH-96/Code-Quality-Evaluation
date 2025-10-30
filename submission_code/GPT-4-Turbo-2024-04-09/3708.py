class Solution:
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        result = []
        rows = len(grid)
        cols = len(grid[0])
        
        for i in range(rows):
            if i % 2 == 0:
                # Traverse from left to right on even rows
                for j in range(0, cols, 2):
                    result.append(grid[i][j])
            else:
                # Traverse from right to left on odd rows
                for j in range(cols - 1 if cols % 2 == 0 else cols - 2, -1, -2):
                    result.append(grid[i][j])
        
        return result