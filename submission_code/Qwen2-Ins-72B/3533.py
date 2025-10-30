class Solution:
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        pos = 0
        for command in commands:
            if command == "RIGHT":
                pos += 1
                if pos % n == 0:
                    pos -= 1
            elif command == "LEFT":
                pos -= 1
                if pos % n == n - 1:
                    pos += 1
            elif command == "UP":
                pos -= n
            elif command == "DOWN":
                pos += n
        return pos