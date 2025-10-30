class Solution:
    def clearDigits(self, s: str) -> str:
        result = ""
        i = 0
        while i < len(s):
            if s[i].isdigit():
                i += 1
                if i < len(s) and not s[i-1].isdigit():
                    i += 1
            else:
                result += s[i]
                i += 1
        return result