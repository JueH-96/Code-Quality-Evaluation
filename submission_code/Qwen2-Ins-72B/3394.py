class Solution:
    def minEnd(self, n: int, x: int) -> int:
        if n == 1:
            return x
        # Calculate the minimum possible value for the last element
        return x + (1 << n.bit_length()) - 1