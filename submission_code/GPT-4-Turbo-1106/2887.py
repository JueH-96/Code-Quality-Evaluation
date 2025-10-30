class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = 'aeiouAEIOU'
        # Extract vowels and sort them
        sorted_vowels = sorted([char for char in s if char in vowels])
        
        # Reconstruct the string with sorted vowels
        result = []
        vowel_index = 0
        for char in s:
            if char in vowels:
                result.append(sorted_vowels[vowel_index])
                vowel_index += 1
            else:
                result.append(char)
        return ''.join(result)