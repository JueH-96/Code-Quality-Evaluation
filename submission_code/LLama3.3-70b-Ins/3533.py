from typing import List

class Solution:
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        # Initialize the current position of the snake
        current_position = 0
        
        # Define the possible movements
        movements = {
            "UP": -n,
            "RIGHT": 1,
            "DOWN": n,
            "LEFT": -1
        }
        
        # Iterate over each command
        for command in commands:
            # Calculate the new position
            new_position = current_position + movements[command]
            
            # Check if the new position is within the same row
            if (new_position // n) == (current_position // n):
                # Update the current position
                current_position = new_position
            else:
                # If the new position is not in the same row, check if it's moving up or down
                if command == "UP":
                    # If moving up, the new position is the last cell in the previous row
                    current_position = (current_position // n) * n + n - 1
                elif command == "DOWN":
                    # If moving down, the new position is the first cell in the next row
                    current_position = ((current_position // n) + 1) * n
        
        # Return the final position of the snake
        return current_position