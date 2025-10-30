class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        n = len(word)
        count = 0

        for i in range(n - 4):
            substring = word[i:i+5]
            if vowels.issubset(substring):
                consonants = len(substring) - sum(c in vowels for c in substring)
                if consonants == k:
                    count += 1

        return count