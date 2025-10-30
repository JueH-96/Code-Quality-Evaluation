class Solution:
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        result = []
        rows = len(grid)
        cols = len(grid[0])
        
        for i in range(rows):
            if i % 2 == 0:
                for j in range(0, cols, 2):
                    result.append(grid[i][j])
            else:
                for j in range(cols - 1, -1, -2):
                    result.append(grid[i][j])
        
        return result