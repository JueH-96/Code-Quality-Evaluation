class Solution:
    def isValid(self, word: str) -> bool:
        # Check length requirement
        if len(word) < 3:
            return False
        
        # Check all characters are alphanumeric
        for c in word:
            if not c.isalnum():
                return False
        
        vowels = {'a', 'e', 'i', 'o', 'u'}
        has_vowel = False
        has_consonant = False
        
        for c in word:
            if c.isalpha():
                lower_c = c.lower()
                if lower_c in vowels:
                    has_vowel = True
                else:
                    has_consonant = True
        
        return has_vowel and has_consonant