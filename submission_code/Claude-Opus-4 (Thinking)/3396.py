class Solution:
    def isValid(self, word: str) -> bool:
        # Check minimum length
        if len(word) < 3:
            return False
        
        # Check if it contains only digits and English letters
        for char in word:
            if not (char.isdigit() or char.isalpha()):
                return False
        
        # Define vowels
        vowels = set('aeiouAEIOU')
        
        # Check for at least one vowel and one consonant
        has_vowel = False
        has_consonant = False
        
        for char in word:
            if char.isalpha():
                if char in vowels:
                    has_vowel = True
                else:
                    has_consonant = True
        
        return has_vowel and has_consonant