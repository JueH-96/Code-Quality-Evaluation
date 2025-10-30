class Solution:
    def sortVowels(self, s: str) -> str:
        vowel_set = {'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'}
        vowels = [c for c in s if c in vowel_set]
        sorted_vowels = sorted(vowels)
        vowel_iter = iter(sorted_vowels)
        result = []
        for c in s:
            if c in vowel_set:
                result.append(next(vowel_iter))
            else:
                result.append(c)
        return ''.join(result)