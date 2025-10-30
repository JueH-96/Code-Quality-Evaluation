from typing import List

class Solution:
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        i, j = 0, 0
        for command in commands:
            if command == "RIGHT":
                j = (j + 1) % n
            elif command == "LEFT":
                j = (j - 1) % n
            elif command == "UP":
                i = (i - 1) % n
            elif command == "DOWN":
                i = (i + 1) % n
        return i * n + j