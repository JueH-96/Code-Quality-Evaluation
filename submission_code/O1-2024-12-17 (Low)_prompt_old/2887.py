class Solution:
    def sortVowels(self, s: str) -> str:
        vowels_set = set("aeiouAEIOU")
        
        # Extract all vowels from the string
        vowels = [ch for ch in s if ch in vowels_set]
        # Sort them according to ASCII values
        vowels.sort()
        
        # Reconstruct the result
        result = []
        vowel_index = 0
        
        for ch in s:
            if ch in vowels_set:
                # Place the next sorted vowel
                result.append(vowels[vowel_index])
                vowel_index += 1
            else:
                # Keep the consonant in place
                result.append(ch)
        
        return "".join(result)