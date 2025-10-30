class Solution:
    def minimumLength(self, s: str) -> int:
        s = list(s)
        while len(s) > 0:
            if len(s) == 1:
                return 1
            if s[0] != s[-1]:
                return len(s)
            else:
                ch = s[0]
                while len(s) > 0 and s[0] == ch:
                    s.pop(0)
                while len(s) > 0 and s[-1] == ch:
                    s.pop()
        return 0