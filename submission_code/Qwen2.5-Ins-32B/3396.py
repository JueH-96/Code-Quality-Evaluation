class Solution:
    def isValid(self, word: str) -> bool:
        vowels = set('aeiouAEIOU')
        consonants = set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ')
        digits = set('0123456789')
        
        if len(word) < 3:
            return False
        
        has_vowel = False
        has_consonant = False
        has_invalid_char = False
        
        for char in word:
            if char in vowels:
                has_vowel = True
            elif char in consonants:
                has_consonant = True
            elif char not in digits:
                has_invalid_char = True
        
        return has_vowel and has_consonant and not has_invalid_char