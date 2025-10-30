class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        cycle = 2 * (n - 1)
        rem = k % cycle
        current = 0
        direction = 1
        
        for _ in range(rem):
            current += direction
            if current == 0 or current == n - 1:
                direction *= -1
        
        return current