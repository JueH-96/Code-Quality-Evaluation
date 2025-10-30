class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = set('aeiouAEIOU')
        
        # Extract vowels and sort them
        s_vowels = sorted([c for c in s if c in vowels])
        
        # Convert string to list for modification
        result = list(s)
        vowel_idx = 0
        
        # Replace vowels in original string with sorted vowels
        for i in range(len(s)):
            if s[i] in vowels:
                result[i] = s_vowels[vowel_idx]
                vowel_idx += 1
                
        return ''.join(result)