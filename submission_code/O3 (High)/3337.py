class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        """
        Count substrings of s that start and end with character c.
        If c appears k times in s, each pair of occurrences (including the same index twice)
        defines a valid substring. The total amount is therefore k * (k + 1) // 2.
        """
        k = s.count(c)
        return k * (k + 1) // 2