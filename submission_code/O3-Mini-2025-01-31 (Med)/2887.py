class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = set('aeiouAEIOU')
        # Extract vowels from s in the order they appear.
        vowel_list = [c for c in s if c in vowels]
        # Sort the vowels by their ASCII values.
        vowel_list.sort()
        
        result = []
        vowel_index = 0
        # Construct the new string t.
        for char in s:
            if char in vowels:
                result.append(vowel_list[vowel_index])
                vowel_index += 1
            else:
                result.append(char)
        
        return ''.join(result)