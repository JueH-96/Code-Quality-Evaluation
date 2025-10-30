class Solution:
    def minChanges(self, s: str) -> int:
        # A string is beautiful iff we can partition it into even-length
        # substrings of identical bits.  Equivalently, every adjacent pair
        # (s[2i], s[2i+1]) must be equal.  We just count how many pairs
        # are mismatched: each mismatched pair needs exactly one flip.
        n = len(s)
        ans = 0
        for i in range(0, n, 2):
            if s[i] != s[i+1]:
                ans += 1
        return ans