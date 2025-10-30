class Solution:
    def isValid(self, word: str) -> bool:
        # Check minimum length
        if len(word) < 3:
            return False
        
        vowels = set('aeiou')
        has_vowel = False
        has_consonant = False
        
        for ch in word:
            # Allowed characters: digits or English letters
            if ch.isdigit():
                continue
            elif ch.isalpha():
                if ch.lower() in vowels:
                    has_vowel = True
                else:
                    has_consonant = True
            else:
                # invalid character encountered
                return False
        
        # Must contain at least one vowel and one consonant
        return has_vowel and has_consonant