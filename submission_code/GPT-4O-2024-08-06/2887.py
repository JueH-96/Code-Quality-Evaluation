class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = "aeiouAEIOU"
        # Extract vowels from the string
        vowel_list = [char for char in s if char in vowels]
        # Sort the vowels based on their ASCII values
        vowel_list.sort()
        
        # Create an iterator for the sorted vowels
        vowel_iter = iter(vowel_list)
        
        # Construct the result string
        result = []
        for char in s:
            if char in vowels:
                # Replace vowel with the next sorted vowel
                result.append(next(vowel_iter))
            else:
                # Keep consonant in its original place
                result.append(char)
        
        return ''.join(result)