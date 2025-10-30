class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = []
        vowel_indices = []
        for i, char in enumerate(s):
            if char.lower() in ['a', 'e', 'i', 'o', 'u']:
                vowels.append(char)
                vowel_indices.append(i)
        vowels.sort(key=lambda x: x.lower())
        
        result = list(s)
        for i, index in enumerate(vowel_indices):
            result[index] = vowels[i]
        
        return ''.join(result)