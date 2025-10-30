class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = {'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'}
        vowel_indices = [i for i, c in enumerate(s) if c in vowels]
        vowel_chars = [c for c in s if c in vowels]
        
        vowel_chars.sort()
        
        t = list(s)
        for i, index in enumerate(vowel_indices):
            t[index] = vowel_chars[i]
        
        return ''.join(t)