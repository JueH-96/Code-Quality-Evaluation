class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = set("aeiouAEIOU")
        
        # Extract vowels and sort them
        extracted_vowels = [c for c in s if c in vowels]
        extracted_vowels.sort()  # Sort by ASCII value
        
        # Rebuild the string
        result = []
        vowel_idx = 0
        
        for char in s:
            if char in vowels:
                # Replace with the next sorted vowel
                result.append(extracted_vowels[vowel_idx])
                vowel_idx += 1
            else:
                # Keep consonants in place
                result.append(char)
        
        return ''.join(result)