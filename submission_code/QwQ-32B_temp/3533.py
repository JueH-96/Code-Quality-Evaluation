from typing import List

class Solution:
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        x, y = 0, 0
        for cmd in commands:
            if cmd == "UP":
                x -= 1
            elif cmd == "RIGHT":
                y += 1
            elif cmd == "DOWN":
                x += 1
            elif cmd == "LEFT":
                y -= 1
        return x * n + y