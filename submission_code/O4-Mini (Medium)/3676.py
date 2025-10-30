class Solution:
    def smallestNumber(self, n: int) -> int:
        """
        Return the smallest number x >= n such that the binary representation
        of x consists only of set bits (i.e., x = 2^k - 1 for some k).
        """
        # Start with the smallest all-ones number (1) and keep extending it
        # until it is >= n.
        x = 1
        while x < n:
            # Shift left by 1 bit and add 1 to produce the next all-ones number.
            # e.g., 1 (0b1) -> 3 (0b11) -> 7 (0b111) -> 15 (0b1111), etc.
            x = (x << 1) | 1
        return x