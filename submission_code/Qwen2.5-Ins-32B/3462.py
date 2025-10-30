class Solution:
    def doesAliceWin(self, s: str) -> bool:
        vowels = set('aeiou')
        has_odd_vowels = False
        vowel_count = 0
        
        for char in s:
            if char in vowels:
                vowel_count += 1
        
        if vowel_count % 2 == 1:
            return True
        
        for char in s:
            if char in vowels:
                has_odd_vowels = True
                break
        
        return has_odd_vowels