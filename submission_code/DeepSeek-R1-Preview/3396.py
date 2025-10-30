class Solution:
    def isValid(self, word: str) -> bool:
        if len(word) < 3:
            return False
        
        for c in word:
            if not (c.isdigit() or c.isalpha()):
                return False
        
        has_vowel = False
        has_consonant = False
        vowels = {'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'}
        
        for c in word:
            if c in vowels:
                has_vowel = True
            elif c.isalpha():
                has_consonant = True
        
        return has_vowel and has_consonant