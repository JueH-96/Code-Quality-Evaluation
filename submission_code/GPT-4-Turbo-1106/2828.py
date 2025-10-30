class Solution:
    def smallestString(self, s: str) -> str:
        # Convert the string to a list of characters for easier manipulation
        chars = list(s)
        
        # Iterate through the characters of the string
        for i in range(len(chars)):
            # If the current character is not 'a', we can replace it and all following characters
            # with their previous characters to get a lexicographically smaller string
            if chars[i] != 'a':
                # Replace the current character and all following characters until we hit an 'a'
                while i < len(chars) and chars[i] != 'a':
                    # Replace the character with its previous character
                    chars[i] = chr(ord(chars[i]) - 1)
                    i += 1
                # We only need to perform the operation once, so break after the first replacement
                break
        
        # Convert the list of characters back to a string and return it
        return ''.join(chars)