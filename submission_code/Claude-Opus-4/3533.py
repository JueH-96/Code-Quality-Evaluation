class Solution:
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        # Start at cell 0, which is position (0, 0)
        row, col = 0, 0
        
        # Process each command
        for command in commands:
            if command == "UP":
                row -= 1
            elif command == "DOWN":
                row += 1
            elif command == "LEFT":
                col -= 1
            elif command == "RIGHT":
                col += 1
        
        # Convert final position to cell number
        return row * n + col