class Solution:
    def clearDigits(self, s: str) -> str:
        s = list(s)
        i = 0
        while i < len(s):
            if s[i].isdigit():
                j = i - 1
                while j >= 0 and s[j].isdigit():
                    j -= 1
                if j >= 0:
                    del s[j]
                    if j < i:
                        i -= 1
                del s[i]
                
            else:
                i += 1
        return "".join(s)