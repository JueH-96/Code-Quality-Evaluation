class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = set('aeiou')
        n = len(word)
        count = 0
        for i in range(n):
            vowel_count = 0
            consonant_count = 0
            for j in range(i, n):
                if word[j] in vowels:
                    vowel_count += 1
                else:
                    consonant_count += 1
                if vowel_count == 5 and consonant_count == k:
                    count += 1
        return count