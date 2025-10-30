from typing import List

class Solution:
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        # Start at the top-left position of the grid (row 0, column 0)
        row, col = 0, 0
        
        # Process each command and update the row and column accordingly.
        for command in commands:
            if command == "UP":
                row -= 1
            elif command == "DOWN":
                row += 1
            elif command == "LEFT":
                col -= 1
            elif command == "RIGHT":
                col += 1
        
        # Calculate the final position using the formula: (row * n) + column.
        return (row * n) + col