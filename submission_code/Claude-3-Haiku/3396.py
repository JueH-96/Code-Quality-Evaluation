class Solution:
    def isValid(self, word: str) -> bool:
        # Check if the word has at least 3 characters
        if len(word) < 3:
            return False
        
        # Check if the word contains only digits, English letters, and no other characters
        if not all(char.isalnum() for char in word):
            return False
        
        # Check if the word contains at least one vowel
        vowels = set('aeiouAEIOU')
        if not any(char in vowels for char in word):
            return False
        
        # Check if the word contains at least one consonant
        consonants = set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ')
        if not any(char in consonants for char in word):
            return False
        
        # If all conditions are met, the word is valid
        return True