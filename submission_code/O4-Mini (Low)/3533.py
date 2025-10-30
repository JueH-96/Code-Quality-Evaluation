from typing import List

class Solution:
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        # Start at top-left corner (0, 0)
        row, col = 0, 0
        
        # Process each command
        for cmd in commands:
            if cmd == "UP":
                row -= 1
            elif cmd == "DOWN":
                row += 1
            elif cmd == "LEFT":
                col -= 1
            elif cmd == "RIGHT":
                col += 1
            # Boundaries are guaranteed by problem statement
        
        # Compute the final cell index
        return row * n + col