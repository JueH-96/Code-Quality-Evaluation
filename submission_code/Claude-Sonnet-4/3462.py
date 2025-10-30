class Solution:
    def doesAliceWin(self, s: str) -> bool:
        vowels = set('aeiou')
        
        # Count total vowels in the string
        vowel_count = sum(1 for char in s if char in vowels)
        
        # If there are no vowels, Alice cannot make any move and loses
        # If there are vowels, Alice can always win with optimal play
        return vowel_count > 0