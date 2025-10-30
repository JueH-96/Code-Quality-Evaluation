class Solution:
    def reverseDegree(self, s: str) -> int:
        total = 0
        for index, char in enumerate(s, start=1):
            reversed_alphabet_pos = 26 - (ord(char) - ord('a'))
            total += reversed_alphabet_pos * index
        return total