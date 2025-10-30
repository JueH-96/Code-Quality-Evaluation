class Solution:
    def smallestNumber(self, n: int) -> int:
        x = n
        while True:
            # Check if x in binary has only set bits
            if (x & (x + 1)) == 0:
                return x
            x += 1