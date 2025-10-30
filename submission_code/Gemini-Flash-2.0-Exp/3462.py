class Solution:
    def doesAliceWin(self, s: str) -> bool:
        vowels = "aeiou"
        vowel_count = 0
        for char in s:
            if char in vowels:
                vowel_count += 1
        
        if vowel_count == 0:
            return False
        
        if vowel_count % 2 == 1:
            return True
        else:
            return False