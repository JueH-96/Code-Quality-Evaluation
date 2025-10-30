class Solution:
    def maxOperations(self, s: str) -> int:
        count = 0
        i = 0
        while i < len(s) - 1:
            if s[i] == '1' and s[i + 1] == '0':
                count += 1
                s = s[:i] + s[i+2:] + s[i]
                i -= 1
                if i < 0:
                    i = 0
            else:
                i += 1
        return count