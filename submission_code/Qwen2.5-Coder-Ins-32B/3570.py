class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = set('aeiou')
        n = len(word)
        count = 0

        def has_all_vowels(sub):
            return all(v in sub for v in vowels)

        for i in range(n):
            consonant_count = 0
            vowel_set = set()
            for j in range(i, n):
                if word[j] in vowels:
                    vowel_set.add(word[j])
                else:
                    consonant_count += 1
                
                if consonant_count == k and has_all_vowels(vowel_set):
                    count += 1
                elif consonant_count > k:
                    break

        return count