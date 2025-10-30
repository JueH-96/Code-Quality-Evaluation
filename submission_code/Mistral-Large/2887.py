class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = 'aeiouAEIOU'
        # Extract vowels from the string
        extracted_vowels = [char for char in s if char in vowels]
        # Sort the extracted vowels
        extracted_vowels.sort()

        # Result list to build the final string
        result = []
        vowel_index = 0

        for char in s:
            if char in vowels:
                # Replace the vowel with the sorted vowel
                result.append(extracted_vowels[vowel_index])
                vowel_index += 1
            else:
                # Keep the consonant in its original place
                result.append(char)

        return ''.join(result)