class Solution:
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        # Initial position of the snake
        row, col = 0, 0

        # Define the movement directions
        directions = {
            "UP": (-1, 0),
            "RIGHT": (0, 1),
            "DOWN": (1, 0),
            "LEFT": (0, -1)
        }

        # Process each command
        for command in commands:
            # Get the movement direction
            dr, dc = directions[command]
            # Update the position
            row += dr
            col += dc

        # Calculate the final position
        final_position = row * n + col
        return final_position