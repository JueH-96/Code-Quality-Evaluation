class Solution:
    def minEnd(self, n: int, x: int) -> int:
        # Calculate the minimal y based on n and x
        delta = n - 1
        y = x | delta
        if y > x:
            return y
        else:
            # Find the position of the first unset bit in x
            mask = ~x & ((1 << 31) - 1)
            first_unset_bit = (mask & -mask).bit_length() - 1
            return x | (1 << first_unset_bit)