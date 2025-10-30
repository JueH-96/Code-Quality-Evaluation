class Solution:
    def isValid(self, word: str) -> bool:
        if len(word) < 3:
            return False
        for c in word:
            if not c.isalnum():
                return False
        has_vowel = False
        has_consonant = False
        vowels = {'a', 'e', 'i', 'o', 'u'}
        for c in word:
            if c.isalpha():
                lower_c = c.lower()
                if lower_c in vowels:
                    has_vowel = True
                else:
                    has_consonant = True
        return has_vowel and has_consonant