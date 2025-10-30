class Solution:
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        # Initial position of the snake
        x, y = 0, 0
        
        # Direction vectors for each command
        directions = {
            "UP": (-1, 0),
            "RIGHT": (0, 1),
            "DOWN": (1, 0),
            "LEFT": (0, -1)
        }
        
        # Process each command
        for command in commands:
            dx, dy = directions[command]
            x += dx
            y += dy
        
        # Calculate the final position in the grid
        return x * n + y