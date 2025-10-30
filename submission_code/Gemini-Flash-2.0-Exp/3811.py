class Solution:
    def reverseDegree(self, s: str) -> int:
        reverse_degree = 0
        for i, char in enumerate(s):
            reverse_alphabet_index = ord('z') - ord(char) + 1
            reverse_degree += reverse_alphabet_index * (i + 1)
        return reverse_degree