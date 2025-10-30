from typing import List

class Solution:
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        position = 0
        for command in commands:
            if command == "UP":
                position -= n
            elif command == "RIGHT":
                position += 1
            elif command == "DOWN":
                position += n
            elif command == "LEFT":
                position -= 1
        return position