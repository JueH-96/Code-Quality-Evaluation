class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        # Count how many times character c appears in string s
        k = s.count(c)
        # The total number of substrings that start and end with c
        # is the number of ways to choose two c positions plus single-letter substrings
        return k * (k + 1) // 2