import math

class Solution:
    def smallestNumber(self, n: int) -> int:
        # Calculate k such that 2^k >= n + 1
        k = math.ceil(math.log2(n + 1))
        # Calculate x as 2^k - 1
        x = (1 << k) - 1
        return x