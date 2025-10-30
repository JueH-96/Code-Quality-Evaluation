class Solution:
    def smallestNumber(self, n: int) -> int:
        # Find the smallest number x >= n with only set bits in its binary representation
        while True:
            if (n & (n - 1)) == 0:
                # n is already a number with only set bits
                return n
            n += 1