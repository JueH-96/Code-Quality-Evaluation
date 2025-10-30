class Solution:
    def sortVowels(self, s: str) -> str:
        # Define the vowels
        vowels = 'aeiouAEIOU'
        
        # Separate the vowels from the string and sort them
        sorted_vowels = sorted([char for char in s if char in vowels])
        
        # Initialize an index to keep track of the current vowel
        vowel_index = 0
        
        # Initialize an empty list to store the result
        result = []
        
        # Iterate over the characters in the string
        for char in s:
            # If the character is a vowel, append the next vowel from the sorted list
            if char in vowels:
                result.append(sorted_vowels[vowel_index])
                vowel_index += 1
            # If the character is a consonant, append it as is
            else:
                result.append(char)
        
        # Join the characters in the result list into a string and return it
        return ''.join(result)