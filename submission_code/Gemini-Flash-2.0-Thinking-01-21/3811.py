class Solution:
    def reverseDegree(self, s: str) -> int:
        total_degree = 0
        # Iterate through the string s with 1-based index
        for index, char in enumerate(s, 1):
            # Calculate the 0-indexed position of the character in the standard alphabet ('a' is 0)
            alpha_pos_0_indexed = ord(char) - ord('a')
            
            # Calculate the 1-indexed position of the character in the reversed alphabet ('a' is 26)
            # If standard 0-indexed pos is p, reversed 1-indexed pos is 26 - p
            reversed_alpha_pos_1_indexed = 26 - alpha_pos_0_indexed
            
            # Calculate the product for this character: reversed alphabet position * string index
            product = reversed_alpha_pos_1_indexed * index
            
            # Add the product to the total degree
            total_degree += product
            
        return total_degree