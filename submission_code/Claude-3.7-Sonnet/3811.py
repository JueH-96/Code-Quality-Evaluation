class Solution:
    def reverseDegree(self, s: str) -> int:
        total = 0
        for i, char in enumerate(s):
            # Calculate the position in the reversed alphabet
            # 'a' is 26, 'b' is 25, ..., 'z' is 1
            reversed_pos = 26 - (ord(char) - ord('a'))
            
            # Calculate the position in the string (1-indexed)
            string_pos = i + 1
            
            # Multiply and add to the total
            total += reversed_pos * string_pos
            
        return total