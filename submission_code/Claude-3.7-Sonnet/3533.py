class Solution:
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        position = 0  # Snake starts at cell 0
        
        for command in commands:
            if command == "UP":
                position -= n
            elif command == "DOWN":
                position += n
            elif command == "LEFT":
                position -= 1
            elif command == "RIGHT":
                position += 1
        
        return position