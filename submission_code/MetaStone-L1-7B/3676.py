class Solution:
    def smallestNumber(self, n: int) -> int:
        h = 1 << (n.bit_length() - 1)
        return h * 2 - 1