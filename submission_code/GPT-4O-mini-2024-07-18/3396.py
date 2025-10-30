class Solution:
    def isValid(self, word: str) -> bool:
        if len(word) < 3:
            return False
        
        has_vowel = False
        has_consonant = False
        valid_characters = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        vowels = set("aeiouAEIOU")
        
        for char in word:
            if char in valid_characters:
                if char in vowels:
                    has_vowel = True
                elif char.isalpha():  # It's a letter but not a vowel, hence a consonant
                    has_consonant = True
            else:
                return False  # Invalid character found
        
        return has_vowel and has_consonant