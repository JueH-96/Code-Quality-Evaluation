class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        n = len(s)
        count = 0
        for i in range(n):
            if s[i] == c:
                count += s[i:].count(c)
        return count