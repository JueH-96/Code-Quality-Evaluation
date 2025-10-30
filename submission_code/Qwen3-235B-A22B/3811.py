class Solution:
    def reverseDegree(self, s: str) -> int:
        total = 0
        for i, c in enumerate(s, start=1):
            reversed_pos = 26 - (ord(c) - ord('a'))
            total += reversed_pos * i
        return total