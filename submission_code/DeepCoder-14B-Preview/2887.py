class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = []
        for c in s:
            if c in 'aeiouAEIOU':
                vowels.append(c)
        sorted_vowels = sorted(vowels, key=lambda x: ord(x))
        result = list(s)
        j = 0
        for i in range(len(s)):
            if s[i] in 'aeiouAEIOU':
                result[i] = sorted_vowels[j]
                j += 1
        return ''.join(result)