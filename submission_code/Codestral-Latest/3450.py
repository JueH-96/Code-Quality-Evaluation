class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        # Initialize the starting child and direction
        current_child = 0
        direction = 1  # 1 means moving to the right, -1 means moving to the left

        # Simulate the passing of the ball for k seconds
        for _ in range(k):
            # Move the ball to the next child
            current_child += direction

            # Check if we need to reverse the direction
            if current_child == n - 1 or current_child == 0:
                direction = -direction

        return current_child