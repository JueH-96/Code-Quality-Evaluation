class Solution:
    def smallestNumber(self, n: int) -> int:
        x = n + 1
        if (x & (x - 1)) == 0:
            return x - 1
        else:
            next_power = 1 << x.bit_length()
            return next_power - 1