class Solution:
    def finalString(self, s: str) -> str:
        # Initialize an empty string to represent the text on the screen.
        screen_text = ""
        
        # Iterate through each character in the input string s.
        for char in s:
            # Check if the current character is 'i'.
            if char == 'i':
                # If it's 'i', reverse the current screen_text.
                # String slicing [::-1] creates a reversed copy of the string.
                screen_text = screen_text[::-1]
            else:
                # If it's not 'i', append the character to the end of screen_text.
                # String concatenation results in a new string.
                screen_text += char
        
        # After processing all characters, return the final screen_text.
        return screen_text