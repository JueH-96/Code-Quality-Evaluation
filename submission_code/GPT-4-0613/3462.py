class Solution:
    def doesAliceWin(self, s: str) -> bool:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        vowel_count = sum(1 for char in s if char in vowels)
        return vowel_count % 2 == 1