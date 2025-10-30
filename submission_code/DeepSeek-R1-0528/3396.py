class Solution:
    def isValid(self, word: str) -> bool:
        n = len(word)
        if n < 3:
            return False
        
        has_vowel = False
        has_consonant = False
        vowels = "aeiouAEIOU"
        
        for ch in word:
            if ch.isdigit():
                continue
            if 'a' <= ch <= 'z' or 'A' <= ch <= 'Z':
                if ch in vowels:
                    has_vowel = True
                else:
                    has_consonant = True
            else:
                return False
        
        return has_vowel and has_consonant