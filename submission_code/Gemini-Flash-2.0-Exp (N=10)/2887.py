class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = "aeiouAEIOU"
        vowel_indices = []
        vowel_chars = []
        for i, char in enumerate(s):
            if char in vowels:
                vowel_indices.append(i)
                vowel_chars.append(char)
        
        vowel_chars.sort()
        
        result = list(s)
        for i, index in enumerate(vowel_indices):
            result[index] = vowel_chars[i]
            
        return "".join(result)