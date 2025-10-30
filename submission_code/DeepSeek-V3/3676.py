class Solution:
    def smallestNumber(self, n: int) -> int:
        if n == 0:
            return 1
        x = n
        while True:
            # Check if x has all bits set
            # A number with all bits set is of the form 2^k - 1
            # So, x + 1 should be a power of 2
            if (x + 1) & x == 0:
                return x
            x += 1