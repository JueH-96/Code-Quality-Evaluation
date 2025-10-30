class Solution:
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        # Start at the top-left cell (0,0), which is position 0
        row, col = 0, 0
        
        for cmd in commands:
            if cmd == "UP":
                row -= 1
            elif cmd == "DOWN":
                row += 1
            elif cmd == "LEFT":
                col -= 1
            elif cmd == "RIGHT":
                col += 1
        
        # Convert the final row, col back into the cell position
        return row * n + col