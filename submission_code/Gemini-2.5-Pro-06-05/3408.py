class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        """
        Calculates the number of special letters in a word.
        A letter is special if it appears in both lowercase and uppercase.
        """
        
        # Create a set of unique characters in the word for efficient lookups.
        present_chars = set(word)
        
        # Initialize a counter for the number of special letters.
        special_count = 0
        
        # Iterate through each lowercase letter of the alphabet.
        # A simple way to do this is to iterate through a string literal.
        for char_lower in "abcdefghijklmnopqrstuvwxyz":
            # Determine the corresponding uppercase letter.
            char_upper = char_lower.upper()
            
            # Check if both the lowercase and uppercase forms are present in the word.
            if char_lower in present_chars and char_upper in present_chars:
                # If so, this is a special letter.
                special_count += 1
                
        return special_count