class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        """
        Count how many substrings of s start and end with character c.
        If c appears k times, the number of such substrings is k choose 2 (for i<j)
        plus k (for i==j), which is k*(k+1)//2.
        """
        k = s.count(c)
        return k * (k + 1) // 2