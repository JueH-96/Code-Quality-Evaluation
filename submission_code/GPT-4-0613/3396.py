class Solution:
    def isValid(self, word: str) -> bool:
        if len(word) < 3:
            return False
        vowels = set('aeiouAEIOU')
        consonants = set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ')
        digits = set('0123456789')
        valid_chars = vowels | consonants | digits
        if not set(word).issubset(valid_chars):
            return False
        if not any(char in vowels for char in word):
            return False
        if not any(char in consonants for char in word):
            return False
        return True