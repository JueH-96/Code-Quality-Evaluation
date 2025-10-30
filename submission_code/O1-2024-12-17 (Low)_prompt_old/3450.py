class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        pos = 0           # Starting child holding the ball
        direction = 1     # 1 for moving right, -1 for moving left
        
        for _ in range(k):
            pos += direction
            # If the ball reaches either end, reverse direction
            if pos == 0 or pos == n - 1:
                direction = -direction
        
        return pos