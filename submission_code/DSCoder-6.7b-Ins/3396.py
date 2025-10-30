class Solution:
    def isValid(self, word: str) -> bool:
        if len(word) < 3:
            return False
        if not any(char.isdigit() for char in word):
            return False
        if not any(char.isalpha() for char in word):
            return False
        if not any(char in 'aeiouAEIOU' for char in word):
            return False
        if not any(char in 'bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ' for char in word):
            return False
        return True