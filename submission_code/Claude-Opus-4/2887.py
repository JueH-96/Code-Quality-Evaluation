class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = set('aeiouAEIOU')
        
        # Extract all vowels from the string
        vowel_chars = []
        for char in s:
            if char in vowels:
                vowel_chars.append(char)
        
        # Sort vowels by ASCII value
        vowel_chars.sort()
        
        # Build the result string
        result = []
        vowel_index = 0
        
        for char in s:
            if char in vowels:
                # Replace with sorted vowel
                result.append(vowel_chars[vowel_index])
                vowel_index += 1
            else:
                # Keep consonant in place
                result.append(char)
        
        return ''.join(result)