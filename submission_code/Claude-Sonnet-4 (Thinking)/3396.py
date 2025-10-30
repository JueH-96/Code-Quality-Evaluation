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
            
            # Check for vowel
            if char in vowels:
                has_vowel = True
            # Check for consonant (must be a letter and not a vowel)
            elif char.isalpha():
                has_consonant = True
        
        return has_vowel and has_consonant