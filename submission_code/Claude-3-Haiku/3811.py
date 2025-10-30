class Solution:
    def reverseDegree(self, s: str) -> int:
        reverse_alphabet = {chr(i + 96): 27 - i for i in range(1, 27)}
        reverse_degree = 0
        for i, char in enumerate(s, 1):
            reverse_degree += reverse_alphabet[char] * i
        return reverse_degree