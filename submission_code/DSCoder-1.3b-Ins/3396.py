class Solution:
    def isValid(self, word: str) -> bool:
        if len(word) < 3:
            return False
        if not word.isdigit():
            return False
        vowels = set('aeiouAEIOU')
        consonants = set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ')
        if not any(char in vowels for char in word):
            return False
        if not any(char in consonants for char in word):
            return False
        return True