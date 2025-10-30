class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = "aeiouAEIOU"
        vowel_list = []
        vowel_indices = []
        for i, char in enumerate(s):
            if char in vowels:
                vowel_list.append(char)
                vowel_indices.append(i)
        
        vowel_list.sort()
        
        result = list(s)
        for i, index in enumerate(vowel_indices):
            result[index] = vowel_list[i]
        
        return "".join(result)