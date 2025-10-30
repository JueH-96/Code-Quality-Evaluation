class Solution:
    def minChanges(self, n: int, k: int) -> int:
        # Check if k has any bits that are not set in n
        if (k & (~n)) != 0:
            return -1
        # The number of 1 bits in (n - k) gives the required changes
        return bin(n - k).count('1')