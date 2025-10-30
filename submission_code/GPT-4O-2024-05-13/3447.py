class Solution:
    def clearDigits(self, s: str) -> str:
        s = list(s)
        while any(c.isdigit() for c in s):
            for i in range(len(s)):
                if s[i].isdigit():
                    for j in range(i - 1, -1, -1):
                        if s[j].isalpha():
                            s.pop(i)
                            s.pop(j)
                            break
                    break
        return ''.join(s)