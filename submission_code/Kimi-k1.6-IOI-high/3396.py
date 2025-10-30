class Solution:
    def isValid(self, word: str) -> bool:
        # Check if the length is at least 3
        if len(word) < 3:
            return False
        
        # Check all characters are alphanumeric (digits or letters)
        for c in word:
            if not (c.isdigit() or c.isalpha()):
                return False
        
        # Check for at least one vowel
        vowels = {'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'}
        has_vowel = False
        for c in word:
            if c in vowels:
                has_vowel = True
                break
        if not has_vowel:
            return False
        
        # Check for at least one consonant
        has_consonant = False
        for c in word:
            if c.isalpha() and c not in vowels:
                has_consonant = True
                break
        if not has_consonant:
            return False
        
        return True