from typing import List

class Solution:
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        # Starting position (row, column)
        r, c = 0, 0
        
        # Mapping each command to its row/col delta
        moves = {
            "UP":    (-1,  0),
            "DOWN":  ( 1,  0),
            "LEFT":  ( 0, -1),
            "RIGHT": ( 0,  1)
        }
        
        # Process every command
        for cmd in commands:
            dr, dc = moves[cmd]
            r += dr
            c += dc
            # No boundary check needed – problem guarantees we stay inside grid
        
        # Convert final (row, col) back to cell index
        return r * n + c