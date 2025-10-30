from typing import List

class Solution:
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        """
        Calculates the final position of a snake in an n x n grid.

        The snake's position is represented by a single integer based on its
        (row, col) coordinates: position = row * n + col.
        """
        
        # A dictionary to map command strings to the change in the 1D position.
        # "UP":    row-1 => position-n
        # "DOWN":  row+1 => position+n
        # "LEFT":  col-1 => position-1
        # "RIGHT": col+1 => position+1
        move_map = {
            "UP": -n,
            "DOWN": n,
            "LEFT": -1,
            "RIGHT": 1
        }
        
        # The snake starts at cell 0.
        position = 0
        
        # Iterate through the sequence of commands and update the position.
        for command in commands:
            position += move_map[command]
            
        return position