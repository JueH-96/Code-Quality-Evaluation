class Solution:
    def isValid(self, word: str) -> bool:
        if len(word) < 3:
            return False
        
        vowels = set('aeiouAEIOU')
        consonants = set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ')
        valid_chars = set('0123456789') | vowels | consonants
        
        has_vowel = False
        has_consonant = False
        
        for char in word:
            if char not in valid_chars:
                return False
            if char in vowels:
                has_vowel = True
            if char in consonants:
                has_consonant = True
        
        return has_vowel and has_consonant