class Solution:
    def doesAliceWin(self, s: str) -> bool:
        vowels = set('aeiou')
        count = 0
        for char in s:
            if char in vowels:
                count += 1
        return count > 0