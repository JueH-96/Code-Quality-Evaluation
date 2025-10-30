class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        def is_vowel(ch):
            return ch in 'aeiou'

        n = len(s)
        beautiful_count = 0

        for i in range(n):
            vowels = 0
            consonants = 0
            for j in range(i, n):
                if is_vowel(s[j]):
                    vowels += 1
                else:
                    consonants += 1

                if vowels == consonants and (vowels * consonants) % k == 0:
                    beautiful_count += 1

        return beautiful_count