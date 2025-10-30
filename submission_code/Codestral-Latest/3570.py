class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = set('aeiou')
        n = len(word)
        count = 0

        for i in range(n):
            vowel_count = 0
            consonant_count = 0
            vowel_set = set()

            for j in range(i, n):
                if word[j] in vowels:
                    vowel_set.add(word[j])
                    vowel_count += 1
                else:
                    consonant_count += 1

                if consonant_count > k:
                    break

                if len(vowel_set) == 5 and consonant_count == k:
                    count += 1

        return count