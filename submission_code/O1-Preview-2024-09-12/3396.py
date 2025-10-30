class Solution:
    def isValid(self, word: str) -> bool:
        if len(word) < 3:
            return False
        if not all(c.isalnum() for c in word):
            return False
        vowels = set('aeiouAEIOU')
        vowel_found = False
        consonant_found = False
        for c in word:
            if c.isalpha():
                if c in vowels:
                    vowel_found = True
                else:
                    consonant_found = True
        return vowel_found and consonant_found