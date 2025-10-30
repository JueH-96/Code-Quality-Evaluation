class Solution:
    def smallestNumber(self, n: int) -> int:
        candidate = 1
        while candidate < n:
            candidate = candidate * 2 + 1
        return candidate