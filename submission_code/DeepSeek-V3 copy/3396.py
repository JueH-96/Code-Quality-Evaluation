class Solution:
    def isValid(self, word: str) -> bool:
        # Check if the word length is at least 3
        if len(word) < 3:
            return False
        
        # Check if the word contains only digits and English letters
        for char in word:
            if not (char.isdigit() or char.isalpha()):
                return False
        
        # Check if the word contains at least one vowel
        vowels = set('aeiouAEIOU')
        has_vowel = False
        for char in word:
            if char in vowels:
                has_vowel = True
                break
        if not has_vowel:
            return False
        
        # Check if the word contains at least one consonant
        consonants = set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ')
        has_consonant = False
        for char in word:
            if char in consonants:
                has_consonant = True
                break
        if not has_consonant:
            return False
        
        return True