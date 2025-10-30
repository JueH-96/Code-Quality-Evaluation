class Solution:
    def isValid(self, word: str) -> bool:
        if len(word) < 3:
            return False
        
        has_digit = False
        has_vowel = False
        has_consonant = False
        
        vowels = set('aeiouAEIOU')
        
        for char in word:
            if char.isdigit():
                has_digit = True
            elif char in vowels:
                has_vowel = True
            elif char.isalpha():
                has_consonant = True
        
        return has_digit and has_vowel and has_consonant