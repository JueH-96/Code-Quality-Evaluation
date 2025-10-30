class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = set("aeiouAEIOU")
        # Extract vowels from s
        vowel_list = [c for c in s if c in vowels]
        # Sort vowels by their ASCII value
        vowel_list.sort(key=lambda x: ord(x))
        
        # Build result by replacing vowels with sorted list results
        result = []
        vowel_index = 0
        for c in s:
            if c in vowels:
                result.append(vowel_list[vowel_index])
                vowel_index += 1
            else:
                result.append(c)
        return ''.join(result)