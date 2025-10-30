from typing import List

class Solution:
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        """
        Calculates the final position of a snake in an n x n grid
        after executing a sequence of commands.

        Args:
            n: The size of the n x n grid.
            commands: A list of strings representing the movement commands.

        Returns:
            The final linear position (index) of the snake.
        """
        # The snake starts at cell 0, which is (0, 0) in grid coordinates.
        # The linear index k corresponds to grid[i][j] where i = k // n and j = k % n.
        # Conversely, grid[i][j] corresponds to linear index (i * n) + j.
        
        current_row = 0
        current_col = 0
        
        # Dictionary to map commands to changes in row and column
        movements = {
            "UP": (-1, 0),
            "DOWN": (1, 0),
            "LEFT": (0, -1),
            "RIGHT": (0, 1)
        }
        
        # Process each command
        for command in commands:
            # Get the change in row and column for the current command
            delta_row, delta_col = movements[command]
            
            # Update the current position
            current_row += delta_row
            current_col += delta_col
            
            # The problem guarantees the snake stays within boundaries,
            # so no boundary checks are needed here.
        
        # Convert the final (row, col) position back to the linear index
        final_position = (current_row * n) + current_col
        
        return final_position