class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        position = 0  # Starting with child 0
        direction = 1  # 1 for right, -1 for left
        
        for _ in range(k):
            position += direction
            
            # If we've reached the end, change direction for the next iteration
            if position == n - 1 or position == 0:
                direction *= -1
        
        return position