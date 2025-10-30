class Solution:
    def doesAliceWin(self, s: str) -> bool:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        count = 0
        for c in s:
            if c in vowels:
                count += 1
        if count % 2 == 1:
            return True
        else:
            return count > 0