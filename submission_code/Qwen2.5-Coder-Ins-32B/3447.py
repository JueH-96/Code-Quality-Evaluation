class Solution:
    def clearDigits(self, s: str) -> str:
        s = list(s)
        i = 0
        while i < len(s):
            if s[i].isdigit():
                if i > 0:
                    s.pop(i - 1)
                s.pop(i)
                i -= 1
            else:
                i += 1
        return ''.join(s)