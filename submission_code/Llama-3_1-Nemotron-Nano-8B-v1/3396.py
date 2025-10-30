class Solution:
    def isValid(self, word: str) -> bool:
        if len(word) < 3:
            return False
        for c in word:
            if not c.isalnum():
                return False
        vowels = {'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'}
        has_vowel = any(c in vowels for c in word)
        has_consonant = any(c.isalpha() and c not in vowels for c in word)
        return has_vowel and has_consonant