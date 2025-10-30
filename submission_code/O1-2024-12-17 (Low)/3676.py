class Solution:
    def smallestNumber(self, n: int) -> int:
        # Start from n and increment until we find a number
        # whose binary representation consists only of '1's.
        x = n
        while True:
            # A number whose binary form is all 1-bits has the property x & (x + 1) == 0.
            if x & (x + 1) == 0:
                return x
            x += 1