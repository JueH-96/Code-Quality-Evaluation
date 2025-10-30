class Solution:
    def isValid(self, word: str) -> bool:
        if len(word) < 3:
            return False

        vowels = "aeiouAEIOU"
        has_vowel = False
        has_consonant = False
        
        for char in word:
            if not ('a' <= char <= 'z' or 'A' <= char <= 'Z' or '0' <= char <= '9'):
                return False
            if char in vowels:
                has_vowel = True
            elif 'a' <= char <= 'z' or 'A' <= char <= 'Z':
                has_consonant = True

        return has_vowel and has_consonant