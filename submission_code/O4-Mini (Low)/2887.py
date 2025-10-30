class Solution:
    def sortVowels(self, s: str) -> str:
        # Define the set of vowels (both lowercase and uppercase)
        vowels = set('aeiouAEIOU')
        
        # Extract vowels from s
        vowel_list = [ch for ch in s if ch in vowels]
        # Sort the vowels by their ASCII values
        vowel_list.sort()
        
        # Build result by replacing vowels in order
        res_chars = []
        vi = 0  # index for vowel_list
        for ch in s:
            if ch in vowels:
                res_chars.append(vowel_list[vi])
                vi += 1
            else:
                res_chars.append(ch)
        
        return ''.join(res_chars)