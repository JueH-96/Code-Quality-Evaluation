from typing import List

class Solution:
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        # Initialize the snake's current position (row, col).
        # Cell 0 corresponds to (0, 0) in grid coordinates.
        current_row = 0
        current_col = 0

        # Iterate through each command and update the snake's position.
        for command in commands:
            if command == "UP":
                current_row -= 1
            elif command == "DOWN":
                current_row += 1
            elif command == "LEFT":
                current_col -= 1
            elif command == "RIGHT":
                current_col += 1
            
            # No need for boundary checks as the problem statement guarantees
            # the snake will remain within the grid boundaries.

        # After processing all commands, convert the final (row, col)
        # back to the single integer grid cell index using the formula: (i * n) + j.
        final_position = (current_row * n) + current_col
        
        return final_position