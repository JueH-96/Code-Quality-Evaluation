class Solution:
    def minEnd(self, n: int, x: int) -> int:
        if n == 1:
            return x
        else:
            return (2 ** (x.bit_length() + 1)) - 1