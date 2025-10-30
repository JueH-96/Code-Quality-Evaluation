from typing import List

class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # Check if there is already a 2x2 square of the same color
        for i in range(2):
            for j in range(2):
                if grid[i][j] == grid[i+1][j] == grid[i][j+1] == grid[i+1][j+1]:
                    return True
        
        # Try changing each cell to the opposite color and check if a 2x2 square can be formed
        for i in range(3):
            for j in range(3):
                original_color = grid[i][j]
                grid[i][j] = 'W' if original_color == 'B' else 'B'
                
                # Check all possible 2x2 squares
                for x in range(2):
                    for y in range(2):
                        if grid[x][y] == grid[x+1][y] == grid[x][y+1] == grid[x+1][y+1]:
                            grid[i][j] = original_color
                            return True
                
                grid[i][j] = original_color
        
        # If no 2x2 square can be formed by changing at most one cell, return False
        return False