class Solution:
    def minEnd(self, n: int, x: int) -> int:
        current = x
        for _ in range(n - 1):
            current = (current + 1) | x
        return current