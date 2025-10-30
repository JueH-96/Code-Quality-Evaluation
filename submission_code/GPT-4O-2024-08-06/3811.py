class Solution:
    def reverseDegree(self, s: str) -> int:
        reverse_degree = 0
        for i, char in enumerate(s):
            # Calculate the position in the reversed alphabet
            reversed_position = 26 - (ord(char) - ord('a'))
            # Calculate the product of the reversed position and the 1-indexed position in the string
            product = reversed_position * (i + 1)
            # Add the product to the reverse degree
            reverse_degree += product
        return reverse_degree