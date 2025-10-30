class Solution:
    def doesAliceWin(self, s: str) -> bool:
        vowels = set('aeiou')
        vowel_count = sum(1 for char in s if char in vowels)
        
        # If the total number of vowels is odd, Alice can always win
        return vowel_count % 2 == 1