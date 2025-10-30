class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        position = 0
        direction = 1  # 1 for right, -1 for left
        
        for _ in range(k):
            position += direction
            
            # Check if we need to reverse direction
            if position == n:  # If it goes beyond the last child
                position = n - 1
                direction = -1
            elif position < 0:  # If it goes before the first child
                position = 0
                direction = 1
        
        return position