class Solution:
    def isValid(self, word: str) -> bool:
        vowels = "aeiouAEIOU"
        consonants = "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ"
        
        if len(word) < 3:
            return False
        
        has_vowel = False
        has_consonant = False
        valid_chars = True

        for char in word:
            if not (char.isdigit() or char.isalpha()):
                valid_chars = False
                break
            if char in vowels:
                has_vowel = True
            elif char in consonants:
                has_consonant = True

        return valid_chars and has_vowel and has_consonant