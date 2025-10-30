class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        # Initialize the current child and direction
        current_child = 0
        direction = 1  # 1 for right, -1 for left
        
        # Simulate the passing of the ball for k seconds
        for _ in range(k):
            # Update the current child based on the direction
            current_child += direction
            
            # Check if the current child is at the end of the line
            if current_child == 0 or current_child == n - 1:
                # Reverse the direction
                direction *= -1
        
        return current_child