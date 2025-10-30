class Solution:
    def minEnd(self, n: int, x: int) -> int:
        result = x
        for i in range(1, n):
            result |= result + 1
        return result