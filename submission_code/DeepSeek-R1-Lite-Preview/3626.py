import math

class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        m = n
        while True:
            # Calculate the product of the digits of m
            product = math.prod(int(digit) for digit in str(m))
            # Check if the product is divisible by t
            if product % t == 0:
                return m
            # Increment m and repeat
            m += 1