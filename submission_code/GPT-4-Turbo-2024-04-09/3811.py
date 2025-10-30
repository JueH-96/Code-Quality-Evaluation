class Solution:
    def reverseDegree(self, s: str) -> int:
        # Calculate the reverse degree
        reverse_degree = 0
        for i, char in enumerate(s):
            # Calculate the index in the reversed alphabet
            reversed_alphabet_index = 26 - (ord(char) - ord('a'))
            # Calculate the index in the string (1-indexed)
            string_index = i + 1
            # Calculate the product
            product = reversed_alphabet_index * string_index
            # Add to the total reverse degree
            reverse_degree += product
        return reverse_degree