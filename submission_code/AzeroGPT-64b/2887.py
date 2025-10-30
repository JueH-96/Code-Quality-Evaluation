class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = sorted([c for c in s if c in 'aeiouAEIOU'])
        result = []
        vowel_idx = 0
        for c in s:
            if c in 'aeiouAEIOU':
                result.append(vowels[vowel_idx])
                vowel_idx += 1
            else:
                result.append(c)
        return ''.join(result)