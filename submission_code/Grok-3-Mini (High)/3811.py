class Solution:
    def reverseDegree(self, s: str) -> int:
        return sum((123 - ord(char)) * idx for idx, char in enumerate(s, start=1))