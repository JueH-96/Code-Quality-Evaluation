class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = []
        for c in s:
            if c.lower() in {'a', 'e', 'i', 'o', 'u'}:
                vowels.append(c)
        sorted_vowels = sorted(vowels)
        res = []
        vowel_idx = 0
        for c in s:
            if c.lower() in {'a', 'e', 'i', 'o', 'u'}:
                res.append(sorted_vowels[vowel_idx])
                vowel_idx += 1
            else:
                res.append(c)
        return ''.join(res)