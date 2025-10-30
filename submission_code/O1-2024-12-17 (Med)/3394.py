class Solution:
    def minEnd(self, n: int, x: int) -> int:
        """
        We want an increasing sequence of n positive integers whose bitwise AND is x.
        Each integer must keep all bits of x set (so for every 1-bit in x, that bit
        is 1 in every element). To ensure a 0-bit in x stays 0 in the final AND,
        at least one element must have that bit unset. A simple way to satisfy
        these constraints and get n strictly-increasing values is to pick numbers
        of the form (x OR mask), where 'mask' covers different combinations of the
        0-bits in x.

        In particular, let B = positions of bits that are "0" in x; we can form
        distinct numbers by OR-ing x with different subsets of these 0-bits. To
        get n distinct, strictly-increasing values, we may label those subsets
        by 0, 1, 2, ..., (n-1) (in binary), mapping each binary digit to a distinct
        bit position from B. The last element's mask will be "decode(n-1)" which
        sets exactly those 0-bit positions corresponding to 1-bits in (n-1).

        Hence the minimum possible final element is: x OR decode(n-1).

        This runs in O(log n) time, which is efficient for n <= 1e8.
        """

        # Gather the positions (in ascending order) where x has bit = 0
        zero_bits = []
        for i in range(32):  # 32 bits suffice for x <= 1e8
            if (x & (1 << i)) == 0:
                zero_bits.append(i)

        # We'll construct the mask for (n-1) based on these zero bits
        mask = 0
        r = n - 1
        idx = 0
        while r > 0:
            if (r & 1) == 1:
                mask |= (1 << zero_bits[idx])
            r >>= 1
            idx += 1

        return x | mask