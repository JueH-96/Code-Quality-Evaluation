class Solution:
    def isValid(self, word: str) -> bool:
        if len(word) < 3:
            return False
        has_vowel = False
        has_consonant = False
        for char in word:
            if not ('0' <= char <= '9' or 'a' <= char <= 'z' or 'A' <= char <= 'Z'):
                return False
            if char.lower() in 'aeiou':
                has_vowel = True
            elif 'a' <= char <= 'z' or 'A' <= char <= 'Z':
                has_consonant = True
        if not has_vowel:
            return False
        if not has_consonant:
            return False
        return True