class Solution:
    def clearDigits(self, s: str) -> str:
        while any(char.isdigit() for char in s):
            for i in range(len(s)):
                if s[i].isdigit():
                    left = i - 1
                    while left >= 0 and s[left].isdigit():
                        left -= 1
                    if left >= 0:
                        s = s[:left] + s[left+1:i] + s[i+1:]
                    else:
                        s = s[i+1:]
                    break
        return s