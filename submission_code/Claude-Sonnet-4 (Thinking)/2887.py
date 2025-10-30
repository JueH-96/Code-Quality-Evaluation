class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = set('aeiouAEIOU')
        
        # Extract vowels
        vowel_chars = [char for char in s if char in vowels]
        
        # Sort vowels by ASCII value
        vowel_chars.sort()
        
        # Build result string
        result = []
        vowel_index = 0
        
        for char in s:
            if char in vowels:
                result.append(vowel_chars[vowel_index])
                vowel_index += 1
            else:
                result.append(char)
        
        return ''.join(result)