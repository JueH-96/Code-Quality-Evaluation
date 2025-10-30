class Solution:
    def minimizedStringLength(self, s: str) -> int:
        # Create a set to store unique characters in the string
        unique_chars = set(s)
        
        # Initialize a variable to store the length of the minimized string
        minimized_length = 0
        
        # Iterate over each unique character in the string
        for char in unique_chars:
            # If the count of the character is odd, increment the minimized length by 1
            if s.count(char) % 2 != 0:
                minimized_length += 1
        
        # Return the length of the minimized string
        return minimized_length