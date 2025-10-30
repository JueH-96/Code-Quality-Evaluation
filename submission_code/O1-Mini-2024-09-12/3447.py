class Solution:
    def clearDigits(self, s: str) -> str:
        s = list(s)
        while True:
            for i, c in enumerate(s):
                if c.isdigit():
                    j = i - 1
                    while j >= 0 and s[j].isdigit():
                        j -= 1
                    if j >= 0:
                        # Delete the digit first, then the non-digit
                        s.pop(i)
                        s.pop(j)
                        break
            else:
                break
        return ''.join(s)