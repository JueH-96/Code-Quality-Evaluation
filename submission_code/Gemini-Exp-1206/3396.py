class Solution:
    def isValid(self, word: str) -> bool:
        if len(word) < 3:
            return False
        
        has_vowel = False
        has_consonant = False
        
        for char in word:
            if not ('0' <= char <= '9' or 'a' <= char <= 'z' or 'A' <= char <= 'Z'):
                return False
            
            if char in "aeiouAEIOU":
                has_vowel = True
            elif 'a' <= char <= 'z' or 'A' <= char <= 'Z':
                has_consonant = True
        
        return has_vowel and has_consonant