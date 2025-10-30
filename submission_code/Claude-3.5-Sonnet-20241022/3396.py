class Solution:
    def isValid(self, word: str) -> bool:
        # Check minimum length
        if len(word) < 3:
            return False
            
        # Check for valid characters
        for c in word:
            if not (c.isalpha() or c.isdigit()):
                return False
                
        # Check for vowels and consonants
        vowels = set('aeiouAEIOU')
        has_vowel = False
        has_consonant = False
        
        for c in word:
            if c.isalpha():
                if c in vowels:
                    has_vowel = True
                else:
                    has_consonant = True
                    
        return has_vowel and has_consonant