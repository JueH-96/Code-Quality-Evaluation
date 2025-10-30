class Solution:
    def smallestNumber(self, n: int) -> int:
        if (n & (n + 1)) == 0:
            return n
        m = n.bit_length()
        return (1 << m) - 1