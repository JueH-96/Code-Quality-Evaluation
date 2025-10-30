class Solution:
    def isValid(self, word: str) -> bool:
        if len(word) < 3:
            return False
        
        vowels = set("aeiouAEIOU")
        has_vowel = False
        has_consonant = False
        valid_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        
        for char in word:
            if char not in valid_chars:
                return False
            if char in vowels:
                has_vowel = True
            elif char.isalpha():  # It's a consonant if it's a letter but not a vowel
                has_consonant = True
        
        return has_vowel and has_consonant