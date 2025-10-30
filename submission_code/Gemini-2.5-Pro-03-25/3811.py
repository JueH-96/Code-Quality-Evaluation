import math

class Solution:
    """
    Calculates the reverse degree of a given string s.

    The reverse degree is defined as the sum of products, where each product is 
    formed by multiplying the position of a character in the reversed alphabet 
    ('a'=26, 'b'=25, ..., 'z'=1) with its 1-indexed position in the string.
    """
    def reverseDegree(self, s: str) -> int:
        """
        Calculates the reverse degree of the input string s.

        Args:
            s: The input string, containing only lowercase English letters.

        Returns:
            The calculated reverse degree of the string s.
        """
        
        # Initialize the total reverse degree to 0.
        total_degree = 0
        
        # Iterate through the string using enumerate to get both the 0-based index (i) 
        # and the character (char) at that index.
        for i, char in enumerate(s):
            
            # Calculate the position of the character in the reversed alphabet.
            # 'a' corresponds to 26, 'b' to 25, ..., 'z' to 1.
            # This can be calculated using the ASCII values:
            # ord('z') - ord(char) gives the 0-based distance from 'z' (z=0, y=1, ..., a=25).
            # Adding 1 converts this to the 1-based reversed position (z=1, y=2, ..., a=26).
            # Alternatively, 26 - (ord(char) - ord('a')) also works.
            reversed_alpha_pos = ord('z') - ord(char) + 1
            
            # Calculate the position of the character in the string (1-indexed).
            # Since the loop index `i` is 0-indexed, the 1-indexed position is `i + 1`.
            string_pos = i + 1
            
            # Calculate the product for the current character:
            # (position in reversed alphabet) * (position in string)
            product = reversed_alpha_pos * string_pos
            
            # Add this product to the running total reverse degree.
            total_degree += product
            
        # Return the final calculated reverse degree.
        return total_degree