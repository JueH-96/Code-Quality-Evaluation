class Solution:
    def minEnd(self, n: int, x: int) -> int:
        if x == 0:
            return 0  # Though x is at least 1 per constraints
        d = None
        for k in range(64):
            mask = 1 << k
            if (x & mask) == 0:
                d = mask
                break
        if d is None:
            bits = x.bit_length()
            d = 1 << bits
        return x + (n - 1) * d