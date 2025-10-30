class Solution:
    def reverseDegree(self, s: str) -> int:
        total = 0
        for i in range(len(s)):
            c = s[i]
            val = 26 - (ord(c) - ord('a'))
            total += val * (i + 1)
        return total