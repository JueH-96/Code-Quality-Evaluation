class Solution:
    def minimumLength(self, s: str) -> int:
        while len(s) > 0 and len(set(s)) == 1:
            c = s[0]
            s = s[1:-1]
            if s.count(c) > 1:
                while s.startswith(c) and s.endswith(c):
                    s = s[1:]
            if len(s) == 0:
                return 0
        return len(s)