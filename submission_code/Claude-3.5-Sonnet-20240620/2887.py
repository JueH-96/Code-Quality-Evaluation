class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = 'aeiouAEIOU'
        vowel_indices = []
        vowel_chars = []
        
        # Collect vowels and their indices
        for i, char in enumerate(s):
            if char in vowels:
                vowel_indices.append(i)
                vowel_chars.append(char)
        
        # Sort vowels
        vowel_chars.sort(key=lambda x: ord(x))
        
        # Create a list from the string for easy modification
        result = list(s)
        
        # Replace vowels in the result
        for i, vowel in zip(vowel_indices, vowel_chars):
            result[i] = vowel
        
        # Convert back to string and return
        return ''.join(result)