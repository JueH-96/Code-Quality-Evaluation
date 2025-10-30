class Solution:
    def doesAliceWin(self, s: str) -> bool:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        count = [0, 0]
        for i in range(len(s)):
            if s[i] in vowels:
                count[i % 2] += 1
        return count[0] % 2 == 1 or count[1] % 2 == 0