class Solution:
    def reverseDegree(self, s: str) -> int:
        """
        Calculate the reverse degree of string s.
        For each character c at 1-based position i, its reversed-alphabet value is:
            val(c) = 123 - ord(c)   # 'a' -> 26, 'b' -> 25, ..., 'z' -> 1
        The reverse degree is the sum of i * val(c) over all positions.
        """
        total = 0
        for i, c in enumerate(s, start=1):
            # reversed alphabet index: 'a' -> 26, ..., 'z' -> 1
            rev_idx = 123 - ord(c)
            total += i * rev_idx
        return total