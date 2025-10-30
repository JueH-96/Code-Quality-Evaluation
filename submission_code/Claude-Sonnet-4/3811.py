class Solution:
    def reverseDegree(self, s: str) -> int:
        total = 0
        for i, char in enumerate(s):
            # Position in reversed alphabet: 'a'=26, 'b'=25, ..., 'z'=1
            reversed_alphabet_pos = 27 - (ord(char) - ord('a'))
            # Position in string (1-indexed)
            string_pos = i + 1
            # Add product to total
            total += reversed_alphabet_pos * string_pos
        return total