class Solution:
    def reverseDegree(self, s: str) -> int:
        total_degree = 0
        
        # Iterate through the string with index i
        # The index i is 0-indexed, ranging from 0 to len(s) - 1
        for i in range(len(s)):
            char = s[i]
            
            # Calculate the position of the character in the reversed alphabet
            # 'a' = 26, 'b' = 25, ..., 'z' = 1
            # The standard 0-indexed alphabet position of char is ord(char) - ord('a')
            # This gives 0 for 'a', 1 for 'b', ..., 25 for 'z'
            # The reversed position is 26 minus the standard 0-indexed position
            standard_alpha_pos_0_indexed = ord(char) - ord('a')
            reversed_alpha_pos = 26 - standard_alpha_pos_0_indexed
            
            # Calculate the position of the character in the string (1-indexed)
            # The index i is 0-indexed, so the 1-indexed position is i + 1
            string_pos_1_indexed = i + 1
            
            # Multiply the position in the reversed alphabet by the position in the string
            product = reversed_alpha_pos * string_pos_1_indexed
            
            # Add the product to the total degree
            total_degree += product
            
        return total_degree