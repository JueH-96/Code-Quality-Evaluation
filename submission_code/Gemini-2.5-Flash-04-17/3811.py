class Solution:
    def reverseDegree(self, s: str) -> int:
        total_degree = 0
        # Iterate through the string with its index (0-indexed)
        for index, char in enumerate(s):
            # Calculate the 1-indexed position of the character in the string
            pos_in_string = index + 1
            
            # Calculate the position of the character in the standard alphabet (0-indexed, 'a' is 0, 'b' is 1, ...)
            standard_alpha_pos = ord(char) - ord('a')
            
            # Calculate the position of the character in the reversed alphabet ('a' is 26, 'b' is 25, ...)
            # This is 26 minus the 0-indexed standard position.
            reversed_alpha_pos = 26 - standard_alpha_pos
            
            # Calculate the product for this character
            product = reversed_alpha_pos * pos_in_string
            
            # Add the product to the total degree
            total_degree += product
            
        return total_degree