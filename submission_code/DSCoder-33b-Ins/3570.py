class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        count = 0
        for i in range(len(word)):
            for j in range(i + 5, len(word) + 1):
                if set(word[i:j]).intersection(vowels) == vowels and word[i:j].count('a') + word[i:j].count('e') + word[i:j].count('i') + word[i:j].count('o') + word[i:j].count('u') == k:
                    count += 1
        return count