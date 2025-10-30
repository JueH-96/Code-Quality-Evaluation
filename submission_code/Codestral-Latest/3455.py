class Solution:
    def minimumLength(self, s: str) -> int:
        while True:
            n = len(s)
            i = 1
            while i < n - 1:
                if s[i - 1] == s[i] and s[i] == s[i + 1]:
                    s = s[:i - 1] + s[i + 2:]
                    n -= 2
                    i = 1
                else:
                    i += 1
            if n == len(s):
                break
        return len(s)