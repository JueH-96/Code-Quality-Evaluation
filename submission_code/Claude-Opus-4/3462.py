class Solution:
    def doesAliceWin(self, s: str) -> bool:
        vowels = set('aeiou')
        
        # Check if there's at least one vowel in the string
        for c in s:
            if c in vowels:
                return True
        
        # If no vowels, Alice cannot make any move
        return False