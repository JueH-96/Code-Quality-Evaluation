class Solution:
    def doesAliceWin(self, s: str) -> bool:
        vowels = set('aeiou')
        odd_vowel_count = 0
        
        for char in s:
            if char in vowels:
                odd_vowel_count += 1
        
        # Alice wins if there is an odd number of vowels in the string
        return odd_vowel_count % 2 == 1