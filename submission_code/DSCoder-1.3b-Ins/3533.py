class Solution:
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        snake = [0]
        for command in commands:
            if command == "UP":
                y = snake[-1] // n
                x = snake[-1] % n
                if (x, y-1) not in snake:
                    snake.append((y-1)*n+x)
            elif command == "DOWN":
                y = snake[-1] // n
                x = snake[-1] % n
                if (x, y+1) not in snake:
                    snake.append((y+1)*n+x)
            elif command == "LEFT":
                x = snake[-1] // n
                y = snake[-1] % n
                if (x-1, y) not in snake:
                    snake.append((y)*n+(x-1))
            elif command == "RIGHT":
                x = snake[-1] // n
                y = snake[-1] % n
                if (x+1, y) not in snake:
                    snake.append((y)*n+(x+1))
        return snake[-1]