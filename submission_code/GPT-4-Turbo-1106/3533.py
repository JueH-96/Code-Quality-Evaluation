class Solution:
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        # Initial position of the snake
        x, y = 0, 0
        
        # Dictionary to map commands to movements
        move = {
            "UP": (-1, 0),
            "DOWN": (1, 0),
            "LEFT": (0, -1),
            "RIGHT": (0, 1)
        }
        
        # Execute each command
        for command in commands:
            dx, dy = move[command]
            x, y = x + dx, y + dy
        
        # Calculate the final position index
        return (x * n) + y