class Solution:
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        x, y = 0, 0
        for command in commands:
            if command == "UP":
                y -= 1
            elif command == "DOWN":
                y += 1
            elif command == "LEFT":
                x -= 1
            elif command == "RIGHT":
                x += 1
        return x * n + y