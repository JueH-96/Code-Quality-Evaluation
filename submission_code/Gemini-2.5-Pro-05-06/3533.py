from typing import List

class Solution:
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        
        # The snake starts at cell 0.
        # The formula for a cell's ID at (row, col) is (row * n) + col.
        # For cell 0: (row * n) + col = 0.
        # Since n >= 2 (from constraints), and row, col must be non-negative,
        # this implies row = 0 and col = 0.
        current_row: int = 0
        current_col: int = 0
        
        # Iterate through each command in the sequence.
        # Update the snake's (row, col) coordinates based on the command.
        # The problem guarantees that the snake always stays within the grid boundaries,
        # so no explicit boundary checks (e.g., current_row >= 0 and current_row < n)
        # are needed during movement simulation.
        for command in commands:
            if command == "UP":
                current_row -= 1  # Moving up decreases the row index.
            elif command == "DOWN":
                current_row += 1  # Moving down increases the row index.
            elif command == "LEFT":
                current_col -= 1  # Moving left decreases the column index.
            elif command == "RIGHT":
                current_col += 1  # Moving right increases the column index.
            # According to the problem constraints, 'command' will always be one of these four.
            # Therefore, an 'else' or default case is not necessary.
            
        # After all commands have been executed, calculate the final cell ID.
        # The problem defines grid[i][j] = (i * n) + j, where i is row and j is column.
        # So, final_cell_id = (current_row * n) + current_col.
        final_cell_id: int = (current_row * n) + current_col
        
        return final_cell_id