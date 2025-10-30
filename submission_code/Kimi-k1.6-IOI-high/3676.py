class Solution:
    def smallestNumber(self, n: int) -> int:
        bit_length = n.bit_length()
        return (1 << bit_length) - 1