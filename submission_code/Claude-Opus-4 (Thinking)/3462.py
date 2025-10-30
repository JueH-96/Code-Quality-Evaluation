class Solution:
    def doesAliceWin(self, s: str) -> bool:
        vowels = set('aeiou')
        vowel_count = sum(1 for c in s if c in vowels)
        return vowel_count > 0