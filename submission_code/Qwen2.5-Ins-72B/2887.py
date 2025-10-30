class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = 'AEIOUaeiou'
        vowel_indices = [i for i, char in enumerate(s) if char in vowels]
        sorted_vowels = sorted([s[i] for i in vowel_indices])
        
        t = list(s)
        for i, index in enumerate(vowel_indices):
            t[index] = sorted_vowels[i]
        
        return ''.join(t)