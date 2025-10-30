class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = set(['a', 'e', 'i', 'o', 'u'])
        n = len(word)
        count = 0
        for i in range(n):
            seen_vowels = set()
            consonants = 0
            for j in range(i, n):
                if word[j] in vowels:
                    seen_vowels.add(word[j])
                else:
                    consonants += 1
                if seen_vowels == vowels and consonants == k:
                    count += 1
        return count