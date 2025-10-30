class Solution:
    def reverseDegree(self, s: str) -> int:
        total = 0
        for i in range(1, len(s) + 1):
            c = s[i-1]
            reversed_val = 26 - (ord(c) - ord('a'))
            total += reversed_val * i
        return total