class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        pos = 0
        direction = 1  # Start moving to the right
        
        for _ in range(k):
            pos += direction
            if pos == 0 or pos == n - 1:
                direction *= -1  # Reverse the direction
        
        return pos