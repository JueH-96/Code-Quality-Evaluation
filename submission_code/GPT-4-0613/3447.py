class Solution:
    def clearDigits(self, s: str) -> str:
        while any(char.isdigit() for char in s):
            for i in range(len(s)):
                if s[i].isdigit():
                    s = s[:i-1] + s[i+1:]
                    break
        return s