class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        pos = 0
        direction = 1
        for _ in range(k):
            next_pos = pos + direction
            if next_pos < 0 or next_pos >= n:
                direction *= -1
                next_pos = pos + direction
            pos = next_pos
        return pos