class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
        vowel_chars = [char for char in s if char in vowels]
        sorted_vowels = sorted(vowel_chars, key=ord)

        result = list(s)
        sorted_vowel_index = 0
        for i in range(len(s)):
            if s[i] in vowels:
                result[i] = sorted_vowels[sorted_vowel_index]
                sorted_vowel_index += 1

        return "".join(result)