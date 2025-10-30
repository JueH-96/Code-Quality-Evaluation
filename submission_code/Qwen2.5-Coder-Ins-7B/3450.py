class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        direction = 1
        current = 0
        
        for _ in range(k):
            current += direction
            if current == n:
                current = n - 2
                direction = -1
            elif current == -1:
                current = 1
                direction = 1
        
        return current