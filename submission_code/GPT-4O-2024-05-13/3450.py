class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        position = 0
        direction = 1  # 1 means right, -1 means left
        
        for _ in range(k):
            position += direction
            if position == 0 or position == n - 1:
                direction *= -1
        
        return position