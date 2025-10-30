class Solution:
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        # Start at the initial position (0, 0)
        x, y = 0, 0
        
        # Define the movement for each command
        move = {
            "UP": (-1, 0),
            "DOWN": (1, 0),
            "LEFT": (0, -1),
            "RIGHT": (0, 1)
        }
        
        # Process each command
        for command in commands:
            dx, dy = move[command]
            x += dx
            y += dy
        
        # Calculate the final position in the grid
        final_position = x * n + y
        return final_position