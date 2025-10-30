from typing import List

class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        # Function to check if a 2x2 square has all same colors
        def is_uniform(square):
            return square[0] == square[1] == square[2] == square[3]
        
        # Check all possible 2x2 squares in the 3x3 grid
        for i in range(2):
            for j in range(2):
                # Extract the 2x2 square
                square = [grid[i][j], grid[i][j+1], grid[i+1][j], grid[i+1][j+1]]
                # Check if the square is already uniform
                if is_uniform(square):
                    return True
                # Check if changing one cell can make it uniform
                for k in range(4):
                    original = square[k]
                    square[k] = 'B' if original == 'W' else 'W'
                    if is_uniform(square):
                        return True
                    square[k] = original  # revert the change
        
        return False