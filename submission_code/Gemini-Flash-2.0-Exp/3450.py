class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        pos = 0
        direction = 1  # 1 for right, -1 for left
        for _ in range(k):
            pos += direction
            if pos == n - 1:
                direction = -1
            elif pos == 0:
                direction = 1
        return pos