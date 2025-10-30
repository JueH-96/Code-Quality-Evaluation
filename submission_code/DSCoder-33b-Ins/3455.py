class Solution:
    def minimumLength(self, s: str) -> int:
        while len(s) > 0 and s[0] == s[-1]:
            s = s.strip(s[0])
        return len(s)