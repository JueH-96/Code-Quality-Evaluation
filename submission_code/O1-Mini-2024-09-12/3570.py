class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = set('aeiou')
        count = 0
        n = len(word)
        for i in range(n):
            seen = set()
            consonant_count = 0
            for j in range(i, n):
                if word[j] in vowels:
                    seen.add(word[j])
                else:
                    consonant_count += 1
                if len(seen) == 5 and consonant_count == k:
                    count += 1
        return count