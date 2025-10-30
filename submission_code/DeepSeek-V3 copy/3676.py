class Solution:
    def smallestNumber(self, n: int) -> int:
        if n == 0:
            return 1
        x = 1
        while x < n:
            x = (x << 1) | 1
        return x