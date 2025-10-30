class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        position = 0
        direction = 1  # 1 for right, -1 for left
        
        for _ in range(k):
            position += direction
            if position == n - 1:
                direction = -1
            elif position == 0:
                direction = 1
        
        return position