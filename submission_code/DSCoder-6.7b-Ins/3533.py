class Solution:
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        # Initialize the position of the snake
        position = 0
        
        # Define the directions
        directions = {'UP': -n, 'RIGHT': 1, 'DOWN': n, 'LEFT': -1}
        
        # Execute each command
        for command in commands:
            position += directions[command]
        
        # Return the final position
        return position