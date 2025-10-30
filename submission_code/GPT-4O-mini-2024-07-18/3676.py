class Solution:
    def smallestNumber(self, n: int) -> int:
        # Start from n and keep checking until we find a number with all set bits
        x = n
        while True:
            # Check if x has all bits set
            if (x & (x + 1)) == 0:
                return x
            x += 1