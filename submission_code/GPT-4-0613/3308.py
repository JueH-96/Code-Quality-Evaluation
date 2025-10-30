class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        s = list(s)
        for i in range(26):
            if chr(i + 97) in s:
                s.remove(chr(i + 97))
        return ''.join(s)