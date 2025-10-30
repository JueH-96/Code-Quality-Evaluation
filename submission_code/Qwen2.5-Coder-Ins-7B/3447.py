class Solution:
    def clearDigits(self, s: str) -> str:
        while any(char.isdigit() for char in s):
            for i, char in enumerate(s):
                if char.isdigit():
                    if i == 0:
                        s = s[1:]
                    else:
                        s = s[:i-1] + s[i+1:]
                    break
        return s