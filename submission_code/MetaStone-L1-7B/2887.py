class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = []
        vowel_indices = []
        for i, c in enumerate(s):
            if c.lower() in {'a', 'e', 'i', 'o', 'u'}:
                vowels.append(c)
                vowel_indices.append(i)
        
        vowels_sorted = sorted(vowels, key=lambda x: ord(x))
        
        result = list(s)
        for i in range(len(vowel_indices)):
            result[vowel_indices[i]] = vowels_sorted[i]
        
        return ''.join(result)