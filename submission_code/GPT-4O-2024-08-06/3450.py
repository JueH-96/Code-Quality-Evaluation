class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        # The position of the ball starts at 0
        position = 0
        # The direction is initially towards the right
        direction = 1
        
        for _ in range(k):
            # Move the ball in the current direction
            position += direction
            
            # If the ball reaches the end of the line, reverse the direction
            if position == 0 or position == n - 1:
                direction *= -1
        
        return position