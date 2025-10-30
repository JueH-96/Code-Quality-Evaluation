class Solution:
    def isValid(self, word: str) -> bool:
        # Check minimum length
        if len(word) < 3:
            return False
        
        # Check all characters are alphanumeric
        for c in word:
            if not c.isalnum():
                return False
        
        # Define vowels and check for their presence and consonants
        vowels = {'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'}
        has_vowel = False
        has_consonant = False
        
        for c in word:
            if c in vowels:
                has_vowel = True
            elif c.isalpha():  # This character is a consonant
                has_consonant = True
        
        # Both conditions must be satisfied
        return has_vowel and has_consonant