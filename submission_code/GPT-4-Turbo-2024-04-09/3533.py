class Solution:
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        # Initial position of the snake
        row, col = 0, 0
        
        # Direction vectors for "UP", "RIGHT", "DOWN", "LEFT"
        direction_map = {
            "UP": (-1, 0),
            "RIGHT": (0, 1),
            "DOWN": (1, 0),
            "LEFT": (0, -1)
        }
        
        # Process each command
        for command in commands:
            # Get the direction vector
            dr, dc = direction_map[command]
            # Update the position of the snake
            row += dr
            col += dc
        
        # Convert the final row, col position to the cell number
        return row * n + col