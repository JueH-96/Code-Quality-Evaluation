class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        child = 0
        direction = 1
        
        for _ in range(k):
            child = (child + direction) % n
            if child == 0 or child == n - 1:
                direction *= -1
        
        return child