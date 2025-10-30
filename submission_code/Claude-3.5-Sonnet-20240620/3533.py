class Solution:
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        row, col = 0, 0
        
        for command in commands:
            if command == "UP":
                row = max(0, row - 1)
            elif command == "DOWN":
                row = min(n - 1, row + 1)
            elif command == "LEFT":
                col = max(0, col - 1)
            elif command == "RIGHT":
                col = min(n - 1, col + 1)
        
        return row * n + col