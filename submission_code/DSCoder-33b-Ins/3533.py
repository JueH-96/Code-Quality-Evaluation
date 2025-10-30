class Solution:
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        position = 0
        for command in commands:
            if command == "RIGHT":
                if position % n < n - 1:
                    position += 1
            elif command == "LEFT":
                if position % n > 0:
                    position -= 1
            elif command == "UP":
                if position >= n:
                    position -= n
            elif command == "DOWN":
                if position < n * n - n:
                    position += n
        return position