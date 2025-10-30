class Solution:
    def smallestNumber(self, n: int) -> int:
        b = n.bit_length()
        candidate = (1 << b) - 1
        if candidate >= n:
            return candidate
        return (1 << (b + 1)) - 1