class Solution:
    def doesAliceWin(self, s: str) -> bool:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        has_vowel = any(c in vowels for c in s)
        return has_vowel