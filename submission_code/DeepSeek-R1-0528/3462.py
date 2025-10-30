class Solution:
    def doesAliceWin(self, s: str) -> bool:
        vowels = "aeiou"
        for char in s:
            if char in vowels:
                return True
        return False