class Solution:
    def doesAliceWin(self, s: str) -> bool:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        count = 0
        for char in s:
            if char in vowels:
                count += 1
        # Alice wins if the total number of vowels is odd
        return count % 2 == 1