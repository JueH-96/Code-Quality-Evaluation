class Solution:
    def smallestNumber(self, n: int) -> int:
        m = n.bit_length()
        return (1 << m) - 1