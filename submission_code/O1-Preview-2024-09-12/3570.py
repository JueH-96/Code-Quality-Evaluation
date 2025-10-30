class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        n = len(word)
        total_substrings = 0

        for i in range(n):
            consonant_count = 0
            vowels_found = set()
            for j in range(i, n):
                ch = word[j]
                if ch in vowels:
                    vowels_found.add(ch)
                else:
                    consonant_count += 1
                if consonant_count > k:
                    break
                if len(vowels_found) == 5 and consonant_count == k:
                    total_substrings += 1
        return total_substrings