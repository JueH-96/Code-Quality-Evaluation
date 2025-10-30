class Solution:
    def smallestNumber(self, n: int) -> int:
        # We look for the smallest number of the form (2^k - 1) that is >= n
        # For n <= 1000, checking up to k=10 is sufficient because 2^10 - 1 = 1023
        for i in range(1, 11):
            candidate = (1 << i) - 1  # 2^i - 1
            if candidate >= n:
                return candidate