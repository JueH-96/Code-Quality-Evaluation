class Solution:
    def isValid(self, word: str) -> bool:
        # Condition 1: Check minimum length of 3
        if len(word) < 3:
            return False
        
        vowels = set('aeiouAEIOU')
        has_vowel = False
        has_consonant = False
        
        for char in word:
            # Condition 2: Check if character is only digit or letter
            if not (char.isalpha() or char.isdigit()):
                return False
            
            # Conditions 3 & 4: Check for at least one vowel and one consonant
            if char.isalpha():
                if char in vowels:
                    has_vowel = True
                else:
                    has_consonant = True
        
        # Return true only if both vowel and consonant exist
        return has_vowel and has_consonant