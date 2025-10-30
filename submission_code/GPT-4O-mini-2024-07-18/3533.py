class Solution:
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        # Starting position of the snake
        x, y = 0, 0
        
        # Process each command
        for command in commands:
            if command == "UP":
                x -= 1
            elif command == "DOWN":
                x += 1
            elif command == "LEFT":
                y -= 1
            elif command == "RIGHT":
                y += 1
        
        # Calculate the final position in the grid
        return (x * n) + y