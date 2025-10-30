class Solution:
    def isValid(self, word: str) -> bool:
        if len(word) < 3:
            return False
        
        vowels = {'a', 'e', 'i', 'o', 'u'}
        has_vowel = False
        has_consonant = False
        
        for char in word:
            if not (char.isalpha() or char.isdigit()):
                return False
            if char.isalpha():
                lower_char = char.lower()
                if lower_char in vowels:
                    has_vowel = True
                else:
                    has_consonant = True
        
        return has_vowel and has_consonant