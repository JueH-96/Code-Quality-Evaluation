class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        count = 0
        vowels = ['a', 'e', 'i', 'o', 'u']
        n = len(word)
        for i in range(n):
            for j in range(i, n):
                sub = word[i:j+1]
                all_vowels_present = True
                for vowel in vowels:
                    if vowel not in sub:
                        all_vowels_present = False
                        break
                if all_vowels_present:
                    consonant_count = 0
                    vowel_set = set(vowels)
                    for char in sub:
                        if char not in vowel_set:
                            consonant_count += 1
                    if consonant_count == k:
                        count += 1
        return count