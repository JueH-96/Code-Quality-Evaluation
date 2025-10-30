class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = sorted([c for c in s if c.lower() in 'aeiou'])
        result = ''
        j = 0
        for i in range(len(s)):
            if s[i].lower() in 'aeiou':
                result += vowels[j]
                j += 1
            else:
                result += s[i]
        return result