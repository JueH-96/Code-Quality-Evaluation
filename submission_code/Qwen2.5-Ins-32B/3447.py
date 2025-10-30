class Solution:
    def clearDigits(self, s: str) -> str:
        i = 0
        while i < len(s):
            if s[i].isdigit():
                if i > 0:
                    s = s[:i-1] + s[i+1:]
                else:
                    s = s[i+1:]
                i = max(0, i-1)
            else:
                i += 1
        return s