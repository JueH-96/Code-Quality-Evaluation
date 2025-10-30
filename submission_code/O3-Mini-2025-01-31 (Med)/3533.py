from typing import List

class Solution:
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        # Starting at position 0, which is at row 0 and column 0.
        row, col = 0, 0
        
        # Process each command.
        for command in commands:
            if command == "UP":
                row -= 1
            elif command == "DOWN":
                row += 1
            elif command == "LEFT":
                col -= 1
            elif command == "RIGHT":
                col += 1
                
        # Translate the final (row, col) back to grid index.
        return row * n + col