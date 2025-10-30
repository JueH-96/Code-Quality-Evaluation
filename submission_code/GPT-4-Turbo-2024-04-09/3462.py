class Solution:
    def doesAliceWin(self, s: str) -> bool:
        vowels = set('aeiou')
        total_vowels = sum(1 for char in s if char in vowels)
        
        # Alice wins if the total number of vowels is odd
        return total_vowels % 2 == 1