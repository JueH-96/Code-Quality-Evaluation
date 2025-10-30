class Solution:
    def isValid(self, word: str) -> bool:
        # Check if the word has a minimum of 3 characters
        if len(word) < 3:
            return False
        
        # Initialize flags for vowel and consonant
        has_vowel = False
        has_consonant = False
        
        # Iterate over each character in the word
        for char in word:
            # Check if the character is a digit
            if char.isdigit():
                continue
            
            # Check if the character is an English letter
            if char.isalpha():
                # Check if the character is a vowel
                if char.lower() in 'aeiou':
                    has_vowel = True
                # If the character is not a vowel, it's a consonant
                else:
                    has_consonant = True
            # If the character is not a digit or an English letter, return False
            else:
                return False
        
        # Return True if the word has at least one vowel and one consonant
        return has_vowel and has_consonant