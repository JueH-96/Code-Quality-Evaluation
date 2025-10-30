class Solution:
    def doesAliceWin(self, s: str) -> bool:
        vowels = set('aeiou')
        odd_count = 0
        
        for char in s:
            if char in vowels:
                odd_count = (odd_count + 1) % 2
        
        return odd_count == 1