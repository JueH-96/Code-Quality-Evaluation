class Solution:
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        current_row, current_col = 0, 0
        for cmd in commands:
            if cmd == "UP":
                current_row -= 1
            elif cmd == "DOWN":
                current_row += 1
            elif cmd == "LEFT":
                current_col -= 1
            elif cmd == "RIGHT":
                current_col += 1
        return current_row * n + current_col