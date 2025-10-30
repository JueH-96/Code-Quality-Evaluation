class Solution:
    def isValid(self, word: str) -> bool:
        vowels = set('aeiouAEIOU')
        consonants = set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ')
        has_vowel = False
        has_consonant = False

        if len(word) < 3:
            return False

        for char in word:
            if not char.isalnum():
                return False
            if char in vowels:
                has_vowel = True
            if char in consonants:
                has_consonant = True

        return has_vowel and has_consonant