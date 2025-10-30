class Solution:
    def sortVowels(self, s: str) -> str:
        vowels_set = set("aeiouAEIOU")
        # Extract all vowels from s.
        vowels = [char for char in s if char in vowels_set]
        # Sort the vowels by their ASCII value.
        vowels.sort(key=ord)
        
        result = []
        vowel_index = 0
        # Reconstruct the string: for vowels, use the sorted ones; leave consonants in place.
        for char in s:
            if char in vowels_set:
                result.append(vowels[vowel_index])
                vowel_index += 1
            else:
                result.append(char)
        return "".join(result)