class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        # Initial position of the ball
        position = 0
        # Initial direction of the ball (1 for right, -1 for left)
        direction = 1

        for _ in range(k):
            # Move the ball in the current direction
            position += direction
            # If the ball reaches either end, reverse the direction
            if position == n - 1 or position == 0:
                direction *= -1

        return position