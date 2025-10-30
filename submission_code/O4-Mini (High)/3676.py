class Solution:
    def smallestNumber(self, n: int) -> int:
        k = 0
        # find the smallest k such that (2^k - 1) >= n
        while (1 << k) - 1 < n:
            k += 1
        return (1 << k) - 1