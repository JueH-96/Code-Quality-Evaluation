class Solution:
    def smallestNumber(self, n: int) -> int:
        # Find the smallest k such that 2^k - 1 >= n
        # This means 2^k >= n + 1
        k = 1
        while (1 << k) - 1 < n:
            k += 1
        return (1 << k) - 1