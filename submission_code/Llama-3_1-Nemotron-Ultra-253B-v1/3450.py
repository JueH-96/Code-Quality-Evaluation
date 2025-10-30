class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        current_pos = 0
        direction = 1
        for _ in range(k):
            current_pos += direction
            if current_pos == 0 or current_pos == n - 1:
                direction *= -1
        return current_pos