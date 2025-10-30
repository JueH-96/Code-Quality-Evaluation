class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = set("aeiouAEIOU")
        # Collect all vowels in the string
        collected_vowels = [char for char in s if char in vowels]
        # Sort them by their ASCII values
        collected_vowels.sort()
        
        # We will build the resulting string
        result = []
        # Index to move through sorted vowels
        vowel_idx = 0
        
        for char in s:
            if char in vowels:
                # Place the next vowel from our sorted list
                result.append(collected_vowels[vowel_idx])
                vowel_idx += 1
            else:
                # Consonant remains in original place
                result.append(char)
        
        return "".join(result)