class Solution:
    def isValid(self, word: str) -> bool:
        # Check if the length is at least 3
        if len(word) < 3:
            return False
        
        # Check all characters are alphanumeric
        for c in word:
            if not c.isalnum():
                return False
        
        # Define vowels including uppercase
        vowels = {'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'}
        has_vowel = False
        has_consonant = False
        
        # Check for vowels and consonants
        for c in word:
            if c.isalpha():
                if c in vowels:
                    has_vowel = True
                else:
                    has_consonant = True
        
        # Return True only if both vowel and consonant are present
        return has_vowel and has_consonant