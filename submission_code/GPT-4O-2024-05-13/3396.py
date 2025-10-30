class Solution:
    def isValid(self, word: str) -> bool:
        if len(word) < 3:
            return False
        
        vowels = set("aeiouAEIOU")
        consonants = set("bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ")
        digits = set("0123456789")
        
        has_vowel = False
        has_consonant = False
        
        for char in word:
            if char in vowels:
                has_vowel = True
            elif char in consonants:
                has_consonant = True
            elif char not in digits:
                return False
        
        return has_vowel and has_consonant