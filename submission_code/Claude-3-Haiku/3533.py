class Solution:
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        # Initialize the snake's position
        row, col = 0, 0
        
        # Iterate through the commands
        for command in commands:
            if command == "UP":
                row -= 1
            elif command == "DOWN":
                row += 1
            elif command == "LEFT":
                col -= 1
            else:  # command == "RIGHT"
                col += 1
            
            # Update the snake's position based on the current command
            if row < 0:
                row = n - 1
            elif row >= n:
                row = 0
            if col < 0:
                col = n - 1
            elif col >= n:
                col = 0
        
        # Calculate the final position of the snake
        return row * n + col