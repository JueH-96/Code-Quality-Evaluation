class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        direction = 1
        position = 0
        for _ in range(k):
            position += direction
            if position == 0 or position == n - 1:
                direction *= -1
        return position