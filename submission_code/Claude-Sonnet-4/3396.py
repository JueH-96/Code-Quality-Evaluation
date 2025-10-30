class Solution:
    def isValid(self, word: str) -> bool:
        # Check minimum length
        if len(word) < 3:
            return False
        
        vowels = set('aeiouAEIOU')
        has_vowel = False
        has_consonant = False
        
        for char in word:
            # Check if character is valid (digit or English letter)
            if not (char.isdigit() or char.isalpha()):
                return False
            
            # If it's a letter, check if it's vowel or consonant
            if char.isalpha():
                if char in vowels:
                    has_vowel = True
                else:
                    has_consonant = True
        
        # Must have at least one vowel and one consonant
        return has_vowel and has_consonant