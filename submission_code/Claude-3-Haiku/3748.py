class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        
        # Sort the diagonals in the bottom-left triangle
        for i in range(n-1, -1, -1):
            diagonal = [grid[j][j-i] for j in range(i, n)]
            diagonal.sort(reverse=True)
            for j in range(i, n):
                grid[j][j-i] = diagonal[j-i]
        
        # Sort the diagonals in the top-right triangle
        for i in range(1, n):
            diagonal = [grid[j-i][j] for j in range(i, n)]
            diagonal.sort()
            for j in range(i, n):
                grid[j-i][j] = diagonal[j-i]
        
        return grid