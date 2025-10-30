class Solution:
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        pos = 0
        for cmd in commands:
            if cmd == "UP":
                pos -= n
            elif cmd == "DOWN":
                pos += n
            elif cmd == "LEFT":
                pos -= 1
            elif cmd == "RIGHT":
                pos += 1
        return pos