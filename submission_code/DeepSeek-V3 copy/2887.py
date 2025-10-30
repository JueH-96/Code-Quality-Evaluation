class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = set(['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'])
        vowel_indices = []
        vowel_chars = []
        
        # Identify the indices and characters of vowels in the string
        for i, char in enumerate(s):
            if char in vowels:
                vowel_indices.append(i)
                vowel_chars.append(char)
        
        # Sort the vowels based on their ASCII values
        vowel_chars.sort()
        
        # Build the resulting string
        s_list = list(s)
        for idx, vowel in zip(vowel_indices, vowel_chars):
            s_list[idx] = vowel
        
        return ''.join(s_list)