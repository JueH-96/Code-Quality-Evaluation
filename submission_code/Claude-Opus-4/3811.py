class Solution:
    def reverseDegree(self, s: str) -> int:
        result = 0
        for i, char in enumerate(s):
            # Position in string (1-indexed)
            string_position = i + 1
            
            # Position in reversed alphabet
            # 'a' -> 26, 'b' -> 25, ..., 'z' -> 1
            reversed_alphabet_position = 26 - (ord(char) - ord('a'))
            
            # Add the product to result
            result += reversed_alphabet_position * string_position
        
        return result