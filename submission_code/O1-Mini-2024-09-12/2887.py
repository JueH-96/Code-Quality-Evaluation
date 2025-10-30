class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = set('aeiouAEIOU')
        chars = list(s)
        vowel_chars = sorted([c for c in s if c in vowels])
        vowel_index = 0
        for i in range(len(chars)):
            if chars[i] in vowels:
                chars[i] = vowel_chars[vowel_index]
                vowel_index += 1
        return ''.join(chars)