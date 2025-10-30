class Solution:
    def numSpecial(self, word: str) -> int:
        # Create a set to store unique special characters
        special_chars = set()
        
        # Iterate over each character in the word
        for char in word:
            # Check if the character is in both lowercase and uppercase in the word
            if char.islower() and char.upper() in word:
                # Add the lowercase version of the character to the set
                special_chars.add(char)
            elif char.isupper() and char.lower() in word:
                # Add the lowercase version of the character to the set
                special_chars.add(char.lower())
        
        # Return the number of special characters
        return len(special_chars)