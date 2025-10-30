class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        pos = 0
        direction = 1  # 1 means moving right, -1 means moving left
        
        for _ in range(k):
            pos += direction
            # If we reach the leftmost or rightmost child, reverse direction
            if pos == 0 or pos == n - 1:
                direction = -direction
        
        return pos