class Solution:
    def removeDigits(self, s: str) -> str:
        s = list(s)
        i = 0
        while i < len(s):
            if s[i].isdigit():
                # Find the closest non-digit character to the left
                j = i - 1
                while j >= 0 and s[j].isdigit():
                    j -= 1
                # Remove the first digit and the closest non-digit character to its left
                s.pop(i)
                if j >= 0:
                    s.pop(j)
                # Adjust the index after removing characters
                i -= 1
            i += 1
        return ''.join(s)