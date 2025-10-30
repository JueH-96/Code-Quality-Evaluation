class Solution:
    def doesAliceWin(self, s: str) -> bool:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        current = 0
        for c in s:
            if c in vowels:
                current += 1
            if current % 2 == 1:
                return True
        return current % 2 == 1