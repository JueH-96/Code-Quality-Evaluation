class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        # The ball completes a full cycle (0 to n-1 and back) in 2*(n-1) steps
        cycle_length = 2 * (n - 1)
        
        # Reduce k to avoid unnecessary iterations
        k = k % cycle_length if cycle_length > 0 else k
        
        # Simulate the ball passing
        position = 0
        direction = 1  # 1 for right, -1 for left
        
        for _ in range(k):
            # Move the ball
            position += direction
            
            # Check if we need to reverse direction
            if position == n - 1:
                direction = -1
            elif position == 0:
                direction = 1
                
        return position