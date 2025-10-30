class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = 'aeiouAEIOU'
        s = list(s)
        vowels_in_s = sorted([c for c in s if c in vowels])
        j = 0
        for i in range(len(s)):
            if s[i] in vowels:
                s[i] = vowels_in_s[j]
                j += 1
        return ''.join(s)