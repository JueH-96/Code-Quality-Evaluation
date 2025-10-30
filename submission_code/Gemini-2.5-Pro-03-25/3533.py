import collections
from typing import List

class Solution:
    """
    Solves the snake movement problem on an n x n grid.
    """
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        """
        Calculates the final position ID of the snake after executing commands.

        Args:
            n: The size of the n x n grid.
            commands: A list of strings representing movement commands ("UP", "DOWN", "LEFT", "RIGHT").

        Returns:
            The final cell ID where the snake ends up.
        """
        
        # Start at cell 0, which corresponds to row 0, column 0
        current_row = 0
        current_col = 0
        
        # Process each command
        for command in commands:
            if command == "UP":
                # Move one row up
                current_row -= 1
            elif command == "DOWN":
                # Move one row down
                current_row += 1
            elif command == "LEFT":
                # Move one column left
                current_col -= 1
            elif command == "RIGHT":
                # Move one column right
                current_col += 1
            # No need for boundary checks as the problem guarantees the snake stays within bounds.

        # Calculate the final position ID using the formula: (row * n) + col
        final_position_id = (current_row * n) + current_col
        
        return final_position_id